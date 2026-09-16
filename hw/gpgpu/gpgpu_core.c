/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 *
 * 简化的 RV32IM 指令解释器 + Zicsr (mhartid)，用于 GPU 核心模拟。
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
            uint32_t old = 0;

            /* 目前只支持 mhartid (只读) */
            if (csr == CSR_MHARTID) {
                old = lane->mhartid;
            }
            WRITE_RD(old);
            /* 只读 CSR，忽略写操作 (csrrw/csrrs/csrrc) */
        }
        break;

    default:
        return -1;
    }

    #undef WRITE_RD

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
