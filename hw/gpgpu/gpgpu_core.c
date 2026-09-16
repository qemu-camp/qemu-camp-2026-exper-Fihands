/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 *
 * 简化的 RV32IM + RV32F 指令解释器 + Zicsr (mhartid/fflags/frm/fcsr)，
 * 用于 GPU 核心模拟。
 * 执行模型: 每个 warp 包含 32 个 lane，锁步执行同一条指令。
 * 线程通过 ebreak 结束执行；所有 lane 结束后 warp 完成。
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "gpgpu.h"
#include "gpgpu_core.h"

/* 每个 warp 允许的最大指令数 (防止死循环) */
#define GPGPU_MAX_CYCLES    1000000

/* CTRL 区域大小 (从 GPU 核心视角) */
#define GPGPU_CORE_CTRL_SIZE    0x1000

/* 指令字段提取 */
#define RV_OPCODE(inst)     ((inst) & 0x7F)
#define RV_RD(inst)         (((inst) >> 7) & 0x1F)
#define RV_FUNCT3(inst)     (((inst) >> 12) & 0x7)
#define RV_RS1(inst)        (((inst) >> 15) & 0x1F)
#define RV_RS2(inst)        (((inst) >> 20) & 0x1F)
#define RV_FUNCT7(inst)     (((inst) >> 25) & 0x7F)

/* 立即数解码 (符号扩展) */
static inline int32_t rv_imm_i(uint32_t inst)
{
    return (int32_t)inst >> 20;
}

static inline int32_t rv_imm_s(uint32_t inst)
{
    int32_t hi = (int32_t)inst >> 25;
    return (hi << 5) | ((inst >> 7) & 0x1F);
}

static inline int32_t rv_imm_b(uint32_t inst)
{
    int32_t imm = (((int32_t)inst >> 31) & 0x1) << 12;
    imm |= (((inst) >> 7) & 0x1) << 11;
    imm |= (((inst) >> 25) & 0x3F) << 5;
    imm |= (((inst) >> 8) & 0xF) << 1;
    return imm;
}

static inline int32_t rv_imm_u(uint32_t inst)
{
    return (int32_t)(inst & 0xFFFFF000);
}

static inline int32_t rv_imm_j(uint32_t inst)
{
    int32_t imm = (((int32_t)inst >> 31) & 0x1) << 20;
    imm |= (((inst) >> 12) & 0xFF) << 12;
    imm |= (((inst) >> 20) & 0x1) << 11;
    imm |= (((inst) >> 21) & 0x3FF) << 1;
    return imm;
}

/* rs3 字段提取 (FMADD 系) */
#define RV_RS3(inst)        (((inst) >> 27) & 0x1F)

/* 浮点符号位 */
#define FP32_SIGN_BIT       0x80000000u

/* 低精度浮点扩展 funct7 (自定义，位于 OP-FP 编码空间) */
#define FUNCT7_FCVT_BF16    0x22    /* BF16 转换 */
#define FUNCT7_FCVT_FP8     0x24    /* FP8 E4M3/E5M2 转换 */
#define FUNCT7_FCVT_FP4     0x26    /* FP4 E2M1 转换 */

/* FP32 位模式: 指数全 1 */
#define FP32_EXP_ALL_ONES   0x7F800000u

/* ============================================================================
 * RV32F 浮点辅助
 * ============================================================================
 */

/*
 * gpgpu_fp_set_rm - 校验 RISC-V rm 字段并设置 softfloat 舍入模式
 * rm: 0=RNE 1=RTZ 2=RDN 3=RUP 4=RMM 7=DYN(取 fcsr.frm)
 * 返回: 0 成功，-1 非法 rm (如 DYN 时 frm 为保留值)
 */
static int gpgpu_fp_set_rm(GPGPULane *lane, uint32_t rm, FloatRoundMode *out)
{
    if (rm == 7) {  /* DYN: 使用 fcsr.frm */
        rm = (lane->fcsr >> 5) & 0x7;
    }

    switch (rm) {
    case 0:  /* RNE */
        *out = float_round_nearest_even;
        break;
    case 1:  /* RTZ */
        *out = float_round_to_zero;
        break;
    case 2:  /* RDN */
        *out = float_round_down;
        break;
    case 3:  /* RUP */
        *out = float_round_up;
        break;
    case 4:  /* RMM: round to nearest, ties away from zero */
        *out = float_round_ties_away;
        break;
    default:
        return -1;
    }

    set_float_rounding_mode(*out, &lane->fp_status);
    return 0;
}

/*
 * gpgpu_fp_sync_flags - 把 softfloat 异常标志合并进 lane 的 fflags
 * RISC-V fflags 位序 NX|UF|OF|DZ|NV 与 softfloat 低 5 位一致
 */
static void gpgpu_fp_sync_flags(GPGPULane *lane)
{
    int flags = get_float_exception_flags(&lane->fp_status);

    if (flags) {
        lane->fcsr = (lane->fcsr & ~0x1Fu) | (uint32_t)(flags & 0x1F);
        set_float_exception_flags(0, &lane->fp_status);
    }
}

/*
 * gpgpu_fclass32 - FCLASS.S: 按 RISC-V 分类位序生成类别掩码
 * bit0:-Inf bit1:-normal bit2:-subnormal bit3:-0 bit4:+0
 * bit5:+subnormal bit6:+normal bit7:+Inf bit8:sNaN bit9:qNaN
 */
static uint32_t gpgpu_fclass32(uint32_t f)
{
    bool sign = (f >> 31) != 0;
    uint32_t exp = (f >> 23) & 0xFF;
    uint32_t frac = f & 0x7FFFFF;

    if (exp == 0xFF) {
        if (frac == 0) {
            return sign ? 1u << 0 : 1u << 7;              /* ±Inf */
        }
        return (frac & 0x400000) ? 1u << 9 : 1u << 8;     /* qNaN / sNaN */
    }
    if (exp == 0) {
        if (frac == 0) {
            return sign ? 1u << 3 : 1u << 4;              /* ±0 */
        }
        return sign ? 1u << 2 : 1u << 5;                  /* ±subnormal */
    }
    return sign ? 1u << 1 : 1u << 6;                      /* ±normal */
}

/*
 * float32_to_float4_e2m1 - FP32 -> FP4 E2M1 手写转换 (RNE 舍入 + 饱和)
 *
 * E2M1 (sign 1 + exp 2, bias 1 + mant 1) 可表示的正数值:
 *   000=0  001=0.5  010=1  011=1.5  100=2  101=3  110=4  111=6
 * E2M1 无 Inf/NaN 表示，NaN 与 Inf 输入饱和到 ±6.0，
 * 超出 ±6.0 的有限值同样饱和到 ±6.0。
 *
 * 正数 FP32 位模式单调，故可直接用位模式阈值判断落点。
 * 区间中点为 0.25 / 0.75 / 1.25 / 1.75 / 2.5 / 3.5 / 5.0。
 * 按 RNE 规则: 偶数-奇数边界的中点向下取偶，
 * 奇数-偶数边界的中点向上取偶。
 */
static uint32_t float32_to_float4_e2m1(uint32_t f)
{
    uint32_t sign = f & FP32_SIGN_BIT;
    uint32_t abs_bits = f & ~FP32_SIGN_BIT;
    uint32_t exp = (f >> 23) & 0xFF;
    uint32_t code;

    if (exp == 0xFF) {
        /* NaN / Inf: 饱和到 ±6.0 */
        code = 0x7;
    } else if (abs_bits <= 0x3E800000u) {       /* <= 0.25 (tie->0)   : 0   */
        code = 0x0;
    } else if (abs_bits < 0x3F400000u) {        /* (0.25, 0.75)       : 0.5 */
        code = 0x1;
    } else if (abs_bits <= 0x3FA00000u) {       /* [0.75,1.25 tie->1.0]: 1.0 */
        code = 0x2;
    } else if (abs_bits < 0x3FE00000u) {        /* (1.25, 1.75)       : 1.5 */
        code = 0x3;
    } else if (abs_bits <= 0x40200000u) {       /* [1.75(tie),2.5(tie)]: 2.0 */
        code = 0x4;
    } else if (abs_bits < 0x40600000u) {        /* (2.5, 3.5)         : 3.0 */
        code = 0x5;
    } else if (abs_bits <= 0x40A00000u) {       /* [3.5(tie),5.0(tie)]: 4.0 */
        code = 0x6;
    } else {                                    /* > 5.0              : 6.0 */
        code = 0x7;
    }

    return sign | code;
}

/* ============================================================================
 * Warp 初始化
 * ============================================================================
 */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    memset(warp, 0, sizeof(*warp));

    if (num_threads > GPGPU_WARP_SIZE) {
        num_threads = GPGPU_WARP_SIZE;
    }

    warp->active_mask = (num_threads == GPGPU_WARP_SIZE) ?
                        0xFFFFFFFFu : ((1u << num_threads) - 1);
    warp->thread_id_base = thread_id_base;
    warp->warp_id = warp_id;
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];

    for (uint32_t i = 0; i < GPGPU_WARP_SIZE; i++) {
        GPGPULane *lane = &warp->lanes[i];

        lane->pc = pc;
        lane->active = i < num_threads;
        /* mhartid 位域: [block(19) | warp(8) | tid(5)] */
        lane->mhartid = MHARTID_ENCODE(block_id_linear, warp_id,
                                       thread_id_base + i);

        /* 浮点执行环境: 默认 RNE 舍入 + canonical NaN 模式 */
        set_float_rounding_mode(float_round_nearest_even, &lane->fp_status);
        set_float_exception_flags(0, &lane->fp_status);
        set_default_nan_mode(1, &lane->fp_status);
        /* canonical NaN: 符号位 0, frac 最高位 1 */
        set_float_default_nan_pattern(0b01000000, &lane->fp_status);
        set_snan_bit_is_one(0, &lane->fp_status);
    }
}

/* ============================================================================
 * Lane 内存访问
 * ============================================================================
 * 地址空间 (从 GPU 核心视角):
 *   0x00000000 - 0x7FFFFFFF : VRAM (直接映射)
 *   0x80000000 - 0x80000FFF : CTRL 设备 (线程上下文寄存器)
 */

static uint32_t gpgpu_ctrl_read_reg(GPGPUState *s, GPGPUWarp *warp,
                                    uint32_t lane_idx, uint32_t offset)
{
    uint32_t linear_tid = warp->thread_id_base + lane_idx;
    uint32_t bdx = s->kernel.block_dim[0];
    uint32_t bdy = s->kernel.block_dim[1];
    uint32_t bdx_y = bdx * bdy;
    uint32_t tid_x, tid_y, tid_z;

    if (bdx == 0) {
        bdx = 1;
    }
    if (bdy == 0) {
        bdy = 1;
    }
    bdx_y = bdx * bdy;

    tid_x = linear_tid % bdx;
    tid_y = (linear_tid / bdx) % bdy;
    tid_z = bdx_y ? linear_tid / bdx_y : 0;

    switch (offset) {
    case 0x00:  return tid_x;               /* THREAD_ID_X */
    case 0x04:  return tid_y;               /* THREAD_ID_Y */
    case 0x08:  return tid_z;               /* THREAD_ID_Z */
    case 0x10:  return warp->block_id[0];   /* BLOCK_ID_X */
    case 0x14:  return warp->block_id[1];   /* BLOCK_ID_Y */
    case 0x18:  return warp->block_id[2];   /* BLOCK_ID_Z */
    case 0x20:  return bdx;                 /* BLOCK_DIM_X */
    case 0x24:  return bdy;                 /* BLOCK_DIM_Y */
    case 0x28:  return s->kernel.block_dim[2]; /* BLOCK_DIM_Z */
    case 0x30:  return s->kernel.grid_dim[0];  /* GRID_DIM_X */
    case 0x34:  return s->kernel.grid_dim[1];  /* GRID_DIM_Y */
    case 0x38:  return s->kernel.grid_dim[2];  /* GRID_DIM_Z */
    default:    return 0;
    }
}

static int gpgpu_lane_load(GPGPUState *s, GPGPUWarp *warp, uint32_t lane_idx,
                           uint32_t addr, uint32_t size, uint32_t *val)
{
    /* CTRL 区域: 线程上下文寄存器 */
    if (addr >= GPGPU_CORE_CTRL_BASE &&
        addr < GPGPU_CORE_CTRL_BASE + GPGPU_CORE_CTRL_SIZE) {
        *val = gpgpu_ctrl_read_reg(s, warp, lane_idx,
                                   addr - GPGPU_CORE_CTRL_BASE);
        return 0;
    }

    /* VRAM 区域 */
    if (addr + size > s->vram_size) {
        return -1;
    }

    switch (size) {
    case 1:
        *val = ldub_p(s->vram_ptr + addr);
        break;
    case 2:
        *val = lduw_le_p(s->vram_ptr + addr);
        break;
    case 4:
        *val = ldl_le_p(s->vram_ptr + addr);
        break;
    default:
        return -1;
    }
    return 0;
}

static int gpgpu_lane_store(GPGPUState *s, uint32_t addr, uint32_t size,
                            uint32_t val)
{
    /* CTRL 区域只读，忽略写入 */
    if (addr >= GPGPU_CORE_CTRL_BASE &&
        addr < GPGPU_CORE_CTRL_BASE + GPGPU_CORE_CTRL_SIZE) {
        return 0;
    }

    if (addr + size > s->vram_size) {
        return -1;
    }

    switch (size) {
    case 1:
        stb_p(s->vram_ptr + addr, val);
        break;
    case 2:
        stw_le_p(s->vram_ptr + addr, val);
        break;
    case 4:
        stl_le_p(s->vram_ptr + addr, val);
        break;
    default:
        return -1;
    }
    return 0;
}

/* ============================================================================
 * 单条指令执行 (单个 lane)
 * ============================================================================
 * 返回值: 0 = 继续, 1 = 线程结束 (ebreak/ecall), -1 = 错误
 */

static int gpgpu_exec_insn(GPGPUState *s, GPGPUWarp *warp, uint32_t lane_idx,
                           uint32_t inst)
{
    GPGPULane *lane = &warp->lanes[lane_idx];
    uint32_t pc = lane->pc;
    uint32_t *gpr = lane->gpr;
    uint32_t opcode = RV_OPCODE(inst);
    uint32_t rd = RV_RD(inst);
    uint32_t funct3 = RV_FUNCT3(inst);
    uint32_t rs1 = RV_RS1(inst);
    uint32_t rs2 = RV_RS2(inst);
    uint32_t funct7 = RV_FUNCT7(inst);
    uint32_t v1 = gpr[rs1];
    uint32_t v2 = gpr[rs2];
    uint32_t next_pc = pc + 4;
    uint32_t val = 0;
    int ret;

    /* 写结果 (x0 恒为 0) */
    #define WRITE_RD(v) do { \
        if (rd != 0) { \
            gpr[rd] = (v); \
        } \
    } while (0)

    /* 写浮点结果 (f0 可写，与 x0 不同) */
    #define WRITE_FRD(v) do { \
        lane->fpr[rd] = (v); \
    } while (0)

    switch (opcode) {
    case 0x37:  /* LUI */
        WRITE_RD(rv_imm_u(inst));
        break;

    case 0x17:  /* AUIPC */
        WRITE_RD(pc + rv_imm_u(inst));
        break;

    case 0x6F:  /* JAL */
        WRITE_RD(pc + 4);
        next_pc = pc + rv_imm_j(inst);
        break;

    case 0x67:  /* JALR */
        next_pc = (v1 + rv_imm_i(inst)) & ~1u;
        WRITE_RD(pc + 4);
        break;

    case 0x63:  /* 条件分支 */
        switch (funct3) {
        case 0:  /* BEQ */
            if (v1 == v2) { next_pc = pc + rv_imm_b(inst); }
            break;
        case 1:  /* BNE */
            if (v1 != v2) { next_pc = pc + rv_imm_b(inst); }
            break;
        case 4:  /* BLT */
            if ((int32_t)v1 < (int32_t)v2) { next_pc = pc + rv_imm_b(inst); }
            break;
        case 5:  /* BGE */
            if ((int32_t)v1 >= (int32_t)v2) { next_pc = pc + rv_imm_b(inst); }
            break;
        case 6:  /* BLTU */
            if (v1 < v2) { next_pc = pc + rv_imm_b(inst); }
            break;
        case 7:  /* BGEU */
            if (v1 >= v2) { next_pc = pc + rv_imm_b(inst); }
            break;
        default:
            return -1;
        }
        break;

    case 0x03:  /* LOAD */
        switch (funct3) {
        case 0:  /* LB */
            ret = gpgpu_lane_load(s, warp, lane_idx, v1 + rv_imm_i(inst), 1, &val);
            if (ret) { return -1; }
            WRITE_RD((int32_t)(int8_t)val);
            break;
        case 1:  /* LH */
            ret = gpgpu_lane_load(s, warp, lane_idx, v1 + rv_imm_i(inst), 2, &val);
            if (ret) { return -1; }
            WRITE_RD((int32_t)(int16_t)val);
            break;
        case 2:  /* LW */
            ret = gpgpu_lane_load(s, warp, lane_idx, v1 + rv_imm_i(inst), 4, &val);
            if (ret) { return -1; }
            WRITE_RD(val);
            break;
        case 4:  /* LBU */
            ret = gpgpu_lane_load(s, warp, lane_idx, v1 + rv_imm_i(inst), 1, &val);
            if (ret) { return -1; }
            WRITE_RD(val);
            break;
        case 5:  /* LHU */
            ret = gpgpu_lane_load(s, warp, lane_idx, v1 + rv_imm_i(inst), 2, &val);
            if (ret) { return -1; }
            WRITE_RD(val);
            break;
        default:
            return -1;
        }
        break;

    case 0x23:  /* STORE */
        switch (funct3) {
        case 0:  /* SB */
            ret = gpgpu_lane_store(s, v1 + rv_imm_s(inst), 1, v2);
            break;
        case 1:  /* SH */
            ret = gpgpu_lane_store(s, v1 + rv_imm_s(inst), 2, v2);
            break;
        case 2:  /* SW */
            ret = gpgpu_lane_store(s, v1 + rv_imm_s(inst), 4, v2);
            break;
        default:
            return -1;
        }
        if (ret) { return -1; }
        break;

    case 0x27:  /* STORE-FP */
        switch (funct3) {
        case 2:  /* FSW */
            ret = gpgpu_lane_store(s, v1 + rv_imm_s(inst), 4, lane->fpr[rs2]);
            break;
        default:
            return -1;
        }
        if (ret) { return -1; }
        break;

    case 0x07:  /* LOAD-FP */
        switch (funct3) {
        case 2:  /* FLW */
            ret = gpgpu_lane_load(s, warp, lane_idx, v1 + rv_imm_i(inst),
                                  4, &val);
            if (ret) { return -1; }
            WRITE_FRD(val);
            break;
        default:
            return -1;
        }
        break;

    case 0x13:  /* OP-IMM */
        switch (funct3) {
        case 0:  /* ADDI */
            WRITE_RD(v1 + rv_imm_i(inst));
            break;
        case 2:  /* SLTI */
            WRITE_RD((int32_t)v1 < rv_imm_i(inst) ? 1 : 0);
            break;
        case 3:  /* SLTIU */
            WRITE_RD(v1 < (uint32_t)rv_imm_i(inst) ? 1 : 0);
            break;
        case 4:  /* XORI */
            WRITE_RD(v1 ^ rv_imm_i(inst));
            break;
        case 6:  /* ORI */
            WRITE_RD(v1 | rv_imm_i(inst));
            break;
        case 7:  /* ANDI */
            WRITE_RD(v1 & rv_imm_i(inst));
            break;
        case 1:  /* SLLI */
            if (funct7 != 0x00) { return -1; }
            WRITE_RD(v1 << rs2);
            break;
        case 5:  /* SRLI / SRAI */
            if (funct7 == 0x00) {
                WRITE_RD(v1 >> rs2);
            } else if (funct7 == 0x20) {
                WRITE_RD((uint32_t)((int32_t)v1 >> rs2));
            } else {
                return -1;
            }
            break;
        default:
            return -1;
        }
        break;

    case 0x33:  /* OP */
        if (funct7 == 0x01) {  /* RV32M 乘除法 */
            switch (funct3) {
            case 0:  /* MUL */
                WRITE_RD(v1 * v2);
                break;
            case 1:  /* MULH */
                WRITE_RD((uint32_t)(((int64_t)(int32_t)v1 *
                                     (int64_t)(int32_t)v2) >> 32));
                break;
            case 2:  /* MULHSU */
                WRITE_RD((uint32_t)(((int64_t)(int32_t)v1 *
                                     (uint64_t)v2) >> 32));
                break;
            case 3:  /* MULHU */
                WRITE_RD((uint32_t)(((uint64_t)v1 * (uint64_t)v2) >> 32));
                break;
            case 4:  /* DIV */
                if (v2 == 0) {
                    WRITE_RD(0xFFFFFFFFu);
                } else if (v1 == 0x80000000u && v2 == 0xFFFFFFFFu) {
                    WRITE_RD(0x80000000u);
                } else {
                    WRITE_RD((uint32_t)((int32_t)v1 / (int32_t)v2));
                }
                break;
            case 5:  /* DIVU */
                if (v2 == 0) {
                    WRITE_RD(0xFFFFFFFFu);
                } else {
                    WRITE_RD(v1 / v2);
                }
                break;
            case 6:  /* REM */
                if (v2 == 0) {
                    WRITE_RD(v1);
                } else if (v1 == 0x80000000u && v2 == 0xFFFFFFFFu) {
                    WRITE_RD(0);
                } else {
                    WRITE_RD((uint32_t)((int32_t)v1 % (int32_t)v2));
                }
                break;
            case 7:  /* REMU */
                if (v2 == 0) {
                    WRITE_RD(v1);
                } else {
                    WRITE_RD(v1 % v2);
                }
                break;
            default:
                return -1;
            }
        } else if (funct7 == 0x00 || funct7 == 0x20) {
            switch (funct3) {
            case 0:  /* ADD / SUB */
                WRITE_RD(funct7 == 0x00 ? v1 + v2 : v1 - v2);
                break;
            case 1:  /* SLL */
                if (funct7 != 0x00) { return -1; }
                WRITE_RD(v1 << (v2 & 0x1F));
                break;
            case 2:  /* SLT */
                if (funct7 != 0x00) { return -1; }
                WRITE_RD((int32_t)v1 < (int32_t)v2 ? 1 : 0);
                break;
            case 3:  /* SLTU */
                if (funct7 != 0x00) { return -1; }
                WRITE_RD(v1 < v2 ? 1 : 0);
                break;
            case 4:  /* XOR */
                if (funct7 != 0x00) { return -1; }
                WRITE_RD(v1 ^ v2);
                break;
            case 5:  /* SRL / SRA */
                WRITE_RD(funct7 == 0x00 ? v1 >> (v2 & 0x1F)
                                        : (uint32_t)((int32_t)v1 >> (v2 & 0x1F)));
                break;
            case 6:  /* OR */
                if (funct7 != 0x00) { return -1; }
                WRITE_RD(v1 | v2);
                break;
            case 7:  /* AND */
                if (funct7 != 0x00) { return -1; }
                WRITE_RD(v1 & v2);
                break;
            default:
                return -1;
            }
        } else {
            return -1;
        }
        break;

    case 0x43:  /* FMADD.S */
    case 0x47:  /* FMSUB.S */
    case 0x4B:  /* FNMSUB.S */
    case 0x4F:  /* FNMADD.S */
    {
        uint32_t rs3 = RV_RS3(inst);
        FloatRoundMode rm;
        int mflags;
        float32 res;

        if (funct3 != 0) {  /* RV32F 仅支持 fmt = S */
            return -1;
        }
        if (gpgpu_fp_set_rm(lane, (inst >> 17) & 0x7, &rm)) {
            return -1;
        }

        /* softfloat 语义: round(rs1 * rs2 + rs3) */
        switch (opcode) {
        case 0x43:  /* fmadd */
            mflags = 0;
            break;
        case 0x47:  /* fmsub: rs1*rs2 - rs3 */
            mflags = float_muladd_negate_c;
            break;
        case 0x4B:  /* fnmsub: -(rs1*rs2) + rs3 */
            mflags = float_muladd_negate_product;
            break;
        case 0x4F:  /* fnmadd: -(rs1*rs2 + rs3) */
            mflags = float_muladd_negate_result;
            break;
        default:
            g_assert_not_reached();
        }

        res = float32_muladd(lane->fpr[rs1], lane->fpr[rs2], lane->fpr[rs3],
                             mflags, &lane->fp_status);
        WRITE_FRD(res);
        gpgpu_fp_sync_flags(lane);
        break;
    }

    case 0x53:  /* OP-FP: RV32F 单精度浮点 */
    {
        uint32_t fs1 = lane->fpr[rs1];
        uint32_t fs2 = lane->fpr[rs2];
        FloatRoundMode rm;
        float32 res;
        uint32_t ires;

        /* rm 字段校验并设置舍入模式 (FSQRT/FCVT 的 rm 与其余指令的 fmt 复用 funct3) */
        if (gpgpu_fp_set_rm(lane, funct3, &rm)) {
            return -1;
        }

        switch (funct7) {
        case 0x00:  /* FADD.S */
            res = float32_add(fs1, fs2, &lane->fp_status);
            WRITE_FRD(res);
            gpgpu_fp_sync_flags(lane);
            break;

        case 0x04:  /* FSUB.S */
            res = float32_sub(fs1, fs2, &lane->fp_status);
            WRITE_FRD(res);
            gpgpu_fp_sync_flags(lane);
            break;

        case 0x08:  /* FMUL.S */
            res = float32_mul(fs1, fs2, &lane->fp_status);
            WRITE_FRD(res);
            gpgpu_fp_sync_flags(lane);
            break;

        case 0x0C:  /* FDIV.S */
            res = float32_div(fs1, fs2, &lane->fp_status);
            WRITE_FRD(res);
            gpgpu_fp_sync_flags(lane);
            break;

        case 0x2C:  /* FSQRT.S */
            if (rs2 != 0) {
                return -1;
            }
            res = float32_sqrt(fs1, &lane->fp_status);
            WRITE_FRD(res);
            gpgpu_fp_sync_flags(lane);
            break;

        case 0x10:  /* FSGNJ.S / FSGNJN.S / FSGNJX.S */
            if (funct3 != 0) {
                return -1;
            }
            switch (rs2) {
            case 0:     /* FSGNJ: 注入 fs2 的符号位 */
                res = (fs1 & ~FP32_SIGN_BIT) | (fs2 & FP32_SIGN_BIT);
                break;
            case 1:     /* FSGNJN: 注入 fs2 符号的相反值 */
                res = (fs1 & ~FP32_SIGN_BIT) | (~fs2 & FP32_SIGN_BIT);
                break;
            case 2:     /* FSGNJX: 符号位异或 */
                res = fs1 ^ (fs2 & FP32_SIGN_BIT);
                break;
            default:
                return -1;
            }
            WRITE_FRD(res);
            break;

        case 0x14:  /* FMIN.S / FMAX.S */
            if (funct3 != 0) {
                return -1;
            }
            switch (rs2) {
            case 0:     /* FMIN.S */
                res = float32_minnum(fs1, fs2, &lane->fp_status);
                break;
            case 1:     /* FMAX.S */
                res = float32_maxnum(fs1, fs2, &lane->fp_status);
                break;
            default:
                return -1;
            }
            WRITE_FRD(res);
            gpgpu_fp_sync_flags(lane);
            break;

        case 0x50:  /* FLE.S / FLT.S / FEQ.S */
            if (funct3 != 0) {
                return -1;
            }
            switch (rs2) {
            case 0:     /* FLE.S: 任一 NaN 触发 invalid */
                ires = float32_le(fs1, fs2, &lane->fp_status);
                break;
            case 1:     /* FLT.S */
                ires = float32_lt(fs1, fs2, &lane->fp_status);
                break;
            case 2:     /* FEQ.S: 仅 sNaN 触发 invalid */
                ires = float32_eq_quiet(fs1, fs2, &lane->fp_status);
                break;
            default:
                return -1;
            }
            WRITE_RD(ires);
            gpgpu_fp_sync_flags(lane);
            break;

        case 0x60:  /* FCVT.W.S: f32 -> 有符号整数 */
            if (rs2 != 0) {
                return -1;
            }
            /* softfloat: NaN -> INT32_MAX, 溢出按符号饱和并置 invalid */
            ires = (uint32_t)float32_to_int32(fs1, &lane->fp_status);
            WRITE_RD(ires);
            gpgpu_fp_sync_flags(lane);
            break;

        case 0x61:  /* FCVT.WU.S: f32 -> 无符号整数 */
            if (rs2 != 0) {
                return -1;
            }
            WRITE_RD(float32_to_uint32(fs1, &lane->fp_status));
            gpgpu_fp_sync_flags(lane);
            break;

        case 0x68:  /* FCVT.S.W: 有符号整数 -> f32 */
            if (rs2 != 0) {
                return -1;
            }
            res = int32_to_float32((int32_t)v1, &lane->fp_status);
            WRITE_FRD(res);
            gpgpu_fp_sync_flags(lane);
            break;

        case 0x69:  /* FCVT.S.WU: 无符号整数 -> f32 */
            if (rs2 != 0) {
                return -1;
            }
            res = uint32_to_float32(v1, &lane->fp_status);
            WRITE_FRD(res);
            gpgpu_fp_sync_flags(lane);
            break;

        case 0x70:  /* FMV.X.W / FCLASS.S */
            if (rs2 != 0) {
                return -1;
            }
            switch (funct3) {
            case 0:     /* FMV.X.W: 位透传 */
                WRITE_RD(fs1);
                break;
            case 1:     /* FCLASS.S */
                WRITE_RD(gpgpu_fclass32(fs1));
                break;
            default:
                return -1;
            }
            break;

        case 0x78:  /* FMV.W.X: 位透传 */
            if (funct3 != 0 || rs2 != 0) {
                return -1;
            }
            WRITE_FRD(v1);
            break;

        case FUNCT7_FCVT_BF16:  /* BF16 转换: sign(1)+exp(8)+mant(7) */
            switch (rs2) {
            case 0:     /* FCVT.S.BF16: BF16(低 16 位) -> FP32 */
                res = bfloat16_to_float32(fs1 & 0xFFFFu, &lane->fp_status);
                WRITE_FRD(res);
                gpgpu_fp_sync_flags(lane);
                break;

            case 1:     /* FCVT.BF16.S: FP32 -> BF16 (存低 16 位) */
                WRITE_FRD(float32_to_bfloat16(fs1, &lane->fp_status));
                gpgpu_fp_sync_flags(lane);
                break;

            default:
                return -1;
            }
            break;

        case FUNCT7_FCVT_FP8:   /* FP8 转换: E4M3/E5M2 */
            switch (rs2) {
            case 0:     /* FCVT.S.E4M3: E4M3 -> BF16 -> FP32 */
            {
                bfloat16 bf = float8_e4m3_to_bfloat16(fs1 & 0xFFu,
                                                      &lane->fp_status);
                res = bfloat16_to_float32(bf, &lane->fp_status);
                WRITE_FRD(res);
                gpgpu_fp_sync_flags(lane);
                break;
            }

            case 1:     /* FCVT.E4M3.S: FP32 -> E4M3 (饱和到 ±448) */
                WRITE_FRD(float32_to_float8_e4m3(fs1, true, &lane->fp_status));
                gpgpu_fp_sync_flags(lane);
                break;

            case 2:     /* FCVT.S.E5M2: E5M2 -> BF16 -> FP32 */
            {
                bfloat16 bf = float8_e5m2_to_bfloat16(fs1 & 0xFFu,
                                                      &lane->fp_status);
                res = bfloat16_to_float32(bf, &lane->fp_status);
                WRITE_FRD(res);
                gpgpu_fp_sync_flags(lane);
                break;
            }

            case 3:     /* FCVT.E5M2.S: FP32 -> E5M2 (饱和, Inf 保持) */
                WRITE_FRD(float32_to_float8_e5m2(fs1, true, &lane->fp_status));
                gpgpu_fp_sync_flags(lane);
                break;

            default:
                return -1;
            }
            break;

        case FUNCT7_FCVT_FP4:   /* FP4 E2M1 转换 (4 bit, 饱和到 ±6.0) */
            switch (rs2) {
            case 0:     /* FCVT.S.E2M1: E2M1 -> E4M3 -> BF16 -> FP32 */
            {
                float8_e4m3 e4 = float4_e2m1_to_float8_e4m3(fs1 & 0xFu,
                                                            &lane->fp_status);
                bfloat16 bf = float8_e4m3_to_bfloat16(e4, &lane->fp_status);
                res = bfloat16_to_float32(bf, &lane->fp_status);
                WRITE_FRD(res);
                gpgpu_fp_sync_flags(lane);
                break;
            }

            case 1:     /* FCVT.E2M1.S: FP32 -> E2M1 (阈值舍入 + 饱和) */
                WRITE_FRD(float32_to_float4_e2m1(fs1));
                break;

            default:
                return -1;
            }
            break;

        default:
            return -1;
        }
        break;
    }

    case 0x0F:  /* FENCE */
        break;

    case 0x73:  /* SYSTEM */
        if (inst == 0x00100073) {          /* EBREAK: 线程结束 */
            return 1;
        }
        if (inst == 0x00000073) {          /* ECALL: 视为线程结束 */
            return 1;
        }
        if (inst == 0x10500073) {          /* WFI: 空转 */
            break;
        }
        if (funct3 != 0) {                 /* CSR 指令 */
            uint32_t csr = inst >> 20;
            uint32_t src = (funct3 >= 5) ? rs1 : v1;  /* CSRRW*I 的 src 为 zimm */
            uint32_t old = 0;
            uint32_t new_val = 0;
            bool wok;    /* 是否为可写 CSR */
            bool w;

            /* 读取旧值，确定可写性 */
            switch (csr) {
            case CSR_MHARTID:  /* 只读 */
                old = lane->mhartid;
                wok = false;
                break;
            case CSR_FFLAGS:
                old = lane->fcsr & 0x1F;
                wok = true;
                break;
            case CSR_FRM:
                old = (lane->fcsr >> 5) & 0x7;
                wok = true;
                break;
            case CSR_FCSR:
                old = lane->fcsr & 0xFF;
                wok = true;
                break;
            default:
                old = 0;
                wok = false;
                break;
            }

            /* 计算 CSR 写入值 (funct3 低 2 位: 1=CSRRW 2=CSRRS 3=CSRRC) */
            switch (funct3 & 0x3) {
            case 1:  /* CSRRW(I): 无条件写 */
                w = true;
                new_val = src;
                break;
            case 2:  /* CSRRS(I): src 非 0 才写 */
                w = (src != 0);
                new_val = old | src;
                break;
            case 3:  /* CSRRC(I) */
                w = (src != 0);
                new_val = old & ~src;
                break;
            default:
                return -1;  /* funct3=4 非法 */
            }

            if (wok && w) {
                switch (csr) {
                case CSR_FFLAGS:
                    lane->fcsr = (lane->fcsr & ~0x1Fu) | (new_val & 0x1Fu);
                    break;
                case CSR_FRM:
                    lane->fcsr = (lane->fcsr & ~0xE0u) |
                                 ((new_val & 0x7u) << 5);
                    break;
                case CSR_FCSR:
                    lane->fcsr = new_val & 0xFFu;
                    break;
                default:
                    break;
                }
            }

            WRITE_RD(old);
        }
        break;

    default:
        return -1;
    }

    #undef WRITE_RD
    #undef WRITE_FRD

    lane->pc = next_pc;
    return 0;
}

/* ============================================================================
 * Warp 执行
 * ============================================================================
 */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    uint32_t cycles = 0;

    if (max_cycles == 0) {
        max_cycles = GPGPU_MAX_CYCLES;
    }

    while (warp->active_mask != 0 && cycles < max_cycles) {
        /* 锁步取指: 使用第一个活跃 lane 的 PC */
        uint32_t first = ctz32(warp->active_mask);
        uint32_t pc = warp->lanes[first].pc;
        uint32_t inst;
        int ret;

        /* 取指 (VRAM 边界 + 对齐检查) */
        if ((pc & 0x3) != 0 || pc + 4 > s->vram_size) {
            return -1;
        }
        inst = ldl_le_p(s->vram_ptr + pc);

        /* 对所有处于该 PC 的活跃 lane 执行指令 */
        for (uint32_t i = 0; i < GPGPU_WARP_SIZE; i++) {
            if (!(warp->active_mask & (1u << i))) {
                continue;
            }
            /* 分歧的 lane (PC 不同) 本轮不执行 */
            if (warp->lanes[i].pc != pc) {
                continue;
            }

            ret = gpgpu_exec_insn(s, warp, i, inst);
            if (ret < 0) {
                return -1;
            }
            if (ret == 1) {  /* ebreak: 线程结束 */
                warp->active_mask &= ~(1u << i);
                warp->lanes[i].active = false;
            }
        }

        cycles++;
    }

    /* 所有 lane 结束 = 成功; 超时 = 错误 */
    return (warp->active_mask == 0) ? 0 : -1;
}

/* ============================================================================
 * Kernel 分发与执行
 * ============================================================================
 */
int gpgpu_core_exec_kernel(GPGPUState *s)
{
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t block_threads;
    uint32_t num_warps;
    GPGPUWarp warp;

    for (int i = 0; i < 3; i++) {
        grid_dim[i] = s->kernel.grid_dim[i];
        block_dim[i] = s->kernel.block_dim[i];
        if (grid_dim[i] == 0 || block_dim[i] == 0) {
            return -1;
        }
    }

    /* 内核代码地址必须在 VRAM 范围内 */
    if (s->kernel.kernel_addr + 4 > s->vram_size) {
        return -1;
    }
    uint32_t kernel_pc = (uint32_t)s->kernel.kernel_addr;

    block_threads = (uint64_t)block_dim[0] * block_dim[1] * block_dim[2];
    if (block_threads == 0) {
        return -1;
    }
    num_warps = (uint32_t)((block_threads + GPGPU_WARP_SIZE - 1) /
                           GPGPU_WARP_SIZE);

    /* 遍历 Grid 中的所有 Block */
    for (uint32_t bz = 0; bz < grid_dim[2]; bz++) {
        for (uint32_t by = 0; by < grid_dim[1]; by++) {
            for (uint32_t bx = 0; bx < grid_dim[0]; bx++) {
                uint32_t block_id[3] = { bx, by, bz };
                uint32_t block_linear = bx +
                    grid_dim[0] * (by + grid_dim[1] * bz);

                /* 遍历 Block 中的所有 Warp */
                for (uint32_t w = 0; w < num_warps; w++) {
                    uint32_t tid_base = w * GPGPU_WARP_SIZE;
                    uint32_t num_threads = (uint32_t)MIN(
                        GPGPU_WARP_SIZE, block_threads - tid_base);

                    gpgpu_core_init_warp(&warp, kernel_pc, tid_base,
                                         block_id, num_threads, w,
                                         block_linear);
                    if (gpgpu_core_exec_warp(s, &warp, GPGPU_MAX_CYCLES)) {
                        return -1;
                    }
                }
            }
        }
    }

    return 0;
}
