import os
# os.environ['TRITON_INTERPRET']='1'

import triton
import triton.language as tl
import torch

from triton.testing import do_bench

from natten import use_fused_na, use_autotuner, use_kv_parallelism_in_fused_na
from natten.na2d import na2d

# [B, H, N, N_KEYDIM/N_HEADDIM]
# m: [B, H, N]
@triton.jit
def tiled_swa_fwd_kernel(
    q, k, v, m, o, 
    q_stride0, q_stride1, q_stride2, q_stride3, 
    k_stride0, k_stride1, k_stride2, k_stride3, 
    v_stride0, v_stride1, v_stride2, v_stride3, 
    o_stride0, o_stride1, o_stride2, o_stride3,
    m_stride0, m_stride1, m_stride2,
    sm_scale, N_TILES_ROW, N_TILES_COL, N_WIN_ROW: tl.constexpr, N_WIN_COL: tl.constexpr,
    N_HEAD: tl.constexpr, N_SBLOCK: tl.constexpr, N_KEYDIM: tl.constexpr, N_HEADDIM: tl.constexpr,
    BLOCK_TILE_KV: tl.constexpr, BLOCK_TILE_Q_ROW: tl.constexpr, BLOCK_TILE_Q_COL: tl.constexpr):

    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    BLOCK_TILE_Q: tl.constexpr = BLOCK_TILE_Q_ROW * BLOCK_TILE_Q_COL
    N_WINDOW_WIDTH: tl.constexpr = (2 * N_WIN_COL + BLOCK_TILE_Q_COL)
    N_WINDOW_HEIGHT: tl.constexpr = (2 * N_WIN_ROW + BLOCK_TILE_Q_ROW)
    N_WINDOW_SIZE: tl.constexpr = N_WINDOW_WIDTH * N_WINDOW_HEIGHT

    sm_scale *= 1.44269504 # 1/ln(2)

    BLOCK_ROW = tl.cdiv(N_TILES_ROW, BLOCK_TILE_Q_ROW)
    BLOCK_COL = tl.cdiv(N_TILES_COL, BLOCK_TILE_Q_COL)

    offset_batch = pid_bh // N_HEAD
    offset_head = pid_bh % N_HEAD

    q0_row = BLOCK_TILE_Q_ROW * (pid_q // BLOCK_COL)
    q0_col = BLOCK_TILE_Q_COL * (pid_q % BLOCK_COL)

    win0_row = q0_row - N_WIN_ROW
    win0_col = q0_col - N_WIN_COL
    
    q_bh_offsets = offset_batch * q_stride0 + offset_head * q_stride1
    k_bh_offsets = offset_batch * k_stride0 + offset_head * k_stride1
    v_bh_offsets = offset_batch * v_stride0 + offset_head * v_stride1
    o_bh_offsets = offset_batch * o_stride0 + offset_head * o_stride1
    m_bh_offsets = offset_batch * m_stride0 + offset_head * m_stride1

    q0_row_offsets = q0_row + tl.arange(0, BLOCK_TILE_Q_ROW)
    q0_col_offsets = q0_col + tl.arange(0, BLOCK_TILE_Q_COL)
    q0_grid_offsets = q0_row_offsets[:, None] * N_TILES_COL + q0_col_offsets[None, :]

    sblock_offsets = tl.arange(0, N_SBLOCK)
    key_dim_offsets = tl.arange(0, N_KEYDIM)
    head_dim_offsets = tl.arange(0, N_HEADDIM)

    q_block_ptr = q + q_bh_offsets + (q0_grid_offsets[:,:,None,None] * N_SBLOCK + sblock_offsets[None, None, :, None]) * q_stride2 + key_dim_offsets[None, None, None, :] * q_stride3
    q_block_ptr = tl.reshape(q_block_ptr, (BLOCK_TILE_Q * N_SBLOCK, N_KEYDIM)) # [BLOCK_TILE_Q * N_SBLOCK, N_KEYDIM]

    q_mask = tl.broadcast_to((q0_row_offsets[:, None, None] < N_TILES_ROW) & (q0_col_offsets[None, :, None] < N_TILES_COL), (BLOCK_TILE_Q_ROW, BLOCK_TILE_Q_COL, N_SBLOCK)).reshape((BLOCK_TILE_Q * N_SBLOCK, ))

    m_block_ptr = m + m_bh_offsets + (q0_grid_offsets[:,:,None] * N_SBLOCK + sblock_offsets[None,None,:]) * m_stride2
    m_block_ptr = tl.reshape(m_block_ptr, (BLOCK_TILE_Q * N_SBLOCK, ))

    o_block_ptr = o + o_bh_offsets + (q0_grid_offsets[:,:,None,None] * N_SBLOCK + sblock_offsets[None, None, :, None]) * o_stride2 + head_dim_offsets[None, None, None, :] * o_stride3
    o_block_ptr = tl.reshape(o_block_ptr, (BLOCK_TILE_Q * N_SBLOCK, N_HEADDIM)) # [BLOCK_TILE_Q * N_SBLOCK, N_HEADDIM]

    q_values = tl.load(q_block_ptr, mask = q_mask[:, None]).to(tl.float16) # [BLOCK_TILE_Q * N_SBLOCK, N_KEYDIM]

    q2kv_tile_row_lb = tl.broadcast_to(tl.maximum(q0_row_offsets - N_WIN_ROW, 0)[:, None, None], (BLOCK_TILE_Q_ROW, BLOCK_TILE_Q_COL, N_SBLOCK)).reshape((BLOCK_TILE_Q * N_SBLOCK,)) # [BLOCK_TILE_Q * N_SBLOCK, ]
    q2kv_tile_row_ub = tl.broadcast_to(tl.minimum(q0_row_offsets + N_WIN_ROW, N_TILES_ROW - 1)[:, None, None], (BLOCK_TILE_Q_ROW, BLOCK_TILE_Q_COL, N_SBLOCK)).reshape((BLOCK_TILE_Q * N_SBLOCK,)) # [BLOCK_TILE_Q * N_SBLOCK, ]
    q2kv_tile_col_lb = tl.broadcast_to(tl.maximum(q0_col_offsets - N_WIN_COL, 0)[None, :, None], (BLOCK_TILE_Q_ROW, BLOCK_TILE_Q_COL, N_SBLOCK)).reshape((BLOCK_TILE_Q * N_SBLOCK,)) # [BLOCK_TILE_Q * N_SBLOCK, ]
    q2kv_tile_col_ub = tl.broadcast_to(tl.minimum(q0_col_offsets + N_WIN_COL, N_TILES_COL - 1)[None, :, None], (BLOCK_TILE_Q_ROW, BLOCK_TILE_Q_COL, N_SBLOCK)).reshape((BLOCK_TILE_Q * N_SBLOCK,)) # [BLOCK_TILE_Q * N_SBLOCK, ]

    max_buffer = tl.full((BLOCK_TILE_Q * N_SBLOCK, ), -float("inf"), dtype = tl.float32)
    denom_buffer = tl.full((BLOCK_TILE_Q * N_SBLOCK, ), 1.0, dtype = tl.float32)
    output_buffer = tl.zeros((BLOCK_TILE_Q * N_SBLOCK, N_HEADDIM), dtype = tl.float32)

    BLOCK_TOKENS: tl.constexpr = N_SBLOCK * BLOCK_TILE_KV
    N_WINDOW_TOKENS: tl.constexpr = N_SBLOCK * N_WINDOW_SIZE
    
    for idx in range(0, N_WINDOW_TOKENS, BLOCK_TOKENS):
        kv_idx = idx + tl.arange(0, BLOCK_TOKENS)

        tile_idx = kv_idx // N_SBLOCK
        token_idx = kv_idx % N_SBLOCK

        tile_row = win0_row + (tile_idx // N_WINDOW_WIDTH) # [BLOCK_TOKENS, ]
        tile_col = win0_col + (tile_idx % N_WINDOW_WIDTH) # [BLOCK_TOKENS, ]
        
        attn_mask = ((tile_row[None, :] >= q2kv_tile_row_lb[:, None]) & 
                    (tile_row[None, :] <= q2kv_tile_row_ub[:, None]) & 
                    (tile_col[None, :] >= q2kv_tile_col_lb[:, None]) & 
                    (tile_col[None, :] <= q2kv_tile_col_ub[:, None])).to(tl.int1) # [BLOCK_TILE_Q * N_SBLOCK, BLOCK_TOKENS]
        load_mask = tl.max(attn_mask, 0).to(tl.int1)
    

        kv_token_idx = ((tile_row * N_TILES_COL + tile_col) * N_SBLOCK + token_idx)

        k_offsets = k_bh_offsets + kv_token_idx * k_stride2 # [BLOCK_TOKENS, ]
        v_offsets = v_bh_offsets + kv_token_idx * v_stride2 # [BLOCK_TOKENS, ]

        k_block_ptr = k + k_offsets[None, :] + key_dim_offsets[:, None] * k_stride3 # [N_KEYDIM, BLOCK_TOKENS]
        v_block_ptr = v + v_offsets[:, None] + head_dim_offsets[None, :] * v_stride3 # [BLOCK_TOKENS, N_HEADDIM]

        k_values = tl.load(k_block_ptr, mask = load_mask[None, :]).to(tl.float16)
        qk = tl.dot(q_values, k_values, tl.where(attn_mask, 0, -1e6)) * sm_scale # [BLOCK_TILE_Q * N_SBLOCK, BLOCK_TOKENS]


        max_new = tl.maximum(max_buffer, tl.max(qk, 1))
        qk = qk - max_new[:, None]
        p = tl.exp2(qk) # [BLOCK_TILE_Q * N_SBLOCK, BLOCK_TOKENS]
        denom_new = tl.sum(p, 1)
        alpha = tl.exp2(max_buffer - max_new)

        denom_buffer = denom_buffer * alpha + denom_new
        max_buffer = max_new
        output_buffer = output_buffer * alpha[:, None]
        v_values = tl.load(v_block_ptr, mask = load_mask[:, None]).to(tl.float16)

        output_buffer = tl.dot(p.to(tl.float16), v_values, output_buffer) # [BLOCK_TILE_Q * N_SBLOCK, N_HEADDIM]

    max_buffer += tl.log2(denom_buffer)
    output_buffer = output_buffer / denom_buffer[:, None]

    tl.store(m_block_ptr, max_buffer,  mask = q_mask)
    tl.store(o_block_ptr, output_buffer,  mask = q_mask[:, None])

tiled_swa_fwd_large_configs = [
    triton.Config({'BLOCK_TILE_KV': BKV}, num_stages=s, num_warps=w) \
    for BKV in [1, 2, 4, 8]\
    for s in [3, 4, 7]\
    for w in [4, 8]\
]

@triton.autotune(tiled_swa_fwd_large_configs, key=['N_WIN_ROW', 'N_WIN_COL', 'N_SBLOCK', 'N_KEYDIM', 'N_HEADDIM'])
@triton.jit
def tiled_swa_fwd_large_kernel(
    q, k, v, m, o, 
    q_stride0, q_stride1, q_stride2, q_stride3, 
    k_stride0, k_stride1, k_stride2, k_stride3, 
    v_stride0, v_stride1, v_stride2, v_stride3, 
    o_stride0, o_stride1, o_stride2, o_stride3,
    m_stride0, m_stride1, m_stride2,
    sm_scale, N_TILES_ROW, N_TILES_COL, N_WIN_ROW: tl.constexpr, N_WIN_COL: tl.constexpr,
    N_HEAD: tl.constexpr, N_SBLOCK: tl.constexpr, N_KEYDIM: tl.constexpr, N_HEADDIM: tl.constexpr,
    BLOCK_TILE_KV: tl.constexpr):

    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    N_WINDOW_WIDTH: tl.constexpr = (2 * N_WIN_COL + 1)
    N_WINDOW_HEIGHT: tl.constexpr = (2 * N_WIN_ROW + 1)
    N_WINDOW_SIZE: tl.constexpr = N_WINDOW_WIDTH * N_WINDOW_HEIGHT

    sm_scale *= 1.44269504 # 1/ln(2)

    offset_batch = pid_bh // N_HEAD
    offset_head = pid_bh % N_HEAD

    q0_row = (pid_q // N_TILES_COL)
    q0_col = (pid_q % N_TILES_COL)

    win0_row = q0_row - N_WIN_ROW
    win0_col = q0_col - N_WIN_COL
    
    q_bh_offsets = offset_batch * q_stride0 + offset_head * q_stride1
    k_bh_offsets = offset_batch * k_stride0 + offset_head * k_stride1
    v_bh_offsets = offset_batch * v_stride0 + offset_head * v_stride1
    o_bh_offsets = offset_batch * o_stride0 + offset_head * o_stride1
    m_bh_offsets = offset_batch * m_stride0 + offset_head * m_stride1

    q0_grid_offsets = q0_row * N_TILES_COL + q0_col

    sblock_offsets = tl.arange(0, N_SBLOCK)
    key_dim_offsets = tl.arange(0, N_KEYDIM)
    head_dim_offsets = tl.arange(0, N_HEADDIM)

    q_block_ptr = q + q_bh_offsets + (q0_grid_offsets * N_SBLOCK + sblock_offsets[:, None]) * q_stride2 + key_dim_offsets[None, :] * q_stride3
    # [N_SBLOCK, N_KEYDIM]

    m_block_ptr = m + m_bh_offsets + (q0_grid_offsets * N_SBLOCK + sblock_offsets) * m_stride2
    # [N_SBLOCK,]

    o_block_ptr = o + o_bh_offsets + (q0_grid_offsets * N_SBLOCK + sblock_offsets[:, None]) * o_stride2 + head_dim_offsets[None, :] * o_stride3
    # [N_SBLOCK, N_HEADDIM]

    q_values = tl.load(q_block_ptr).to(tl.float16) # [N_SBLOCK, N_KEYDIM]

    max_buffer = tl.full((N_SBLOCK, ), -float("inf"), dtype = tl.float32)
    denom_buffer = tl.full((N_SBLOCK, ), 1.0, dtype = tl.float32)
    output_buffer = tl.zeros((N_SBLOCK, N_HEADDIM), dtype = tl.float32)

    BLOCK_TOKENS: tl.constexpr = N_SBLOCK * BLOCK_TILE_KV
    N_WINDOW_TOKENS: tl.constexpr = N_SBLOCK * N_WINDOW_SIZE
    
    for idx in range(0, N_WINDOW_TOKENS, BLOCK_TOKENS):
        kv_idx = idx + tl.arange(0, BLOCK_TOKENS)

        tile_idx = kv_idx // N_SBLOCK
        token_idx = kv_idx % N_SBLOCK

        tile_row = win0_row + (tile_idx // N_WINDOW_WIDTH) # [BLOCK_TOKENS, ]
        tile_col = win0_col + (tile_idx % N_WINDOW_WIDTH) # [BLOCK_TOKENS, ]
        
        attn_mask = ((tile_row >= 0) & (tile_row < N_TILES_ROW) & 
                    (tile_col >= 0) & (tile_col < N_TILES_COL) & 
                    (kv_idx < N_WINDOW_TOKENS)) # [BLOCK_TOKENS, ]
    
        kv_token_idx = ((tile_row * N_TILES_COL + tile_col) * N_SBLOCK + token_idx)

        k_offsets = k_bh_offsets + kv_token_idx * k_stride2 # [BLOCK_TOKENS, ]
        v_offsets = v_bh_offsets + kv_token_idx * v_stride2 # [BLOCK_TOKENS, ]

        k_block_ptr = k + k_offsets[None, :] + key_dim_offsets[:, None] * k_stride3 # [N_KEYDIM, BLOCK_TOKENS]
        v_block_ptr = v + v_offsets[:, None] + head_dim_offsets[None, :] * v_stride3 # [BLOCK_TOKENS, N_HEADDIM]

        k_values = tl.load(k_block_ptr, mask = attn_mask[None, :]).to(tl.float16)
        qk = tl.dot(q_values, k_values, tl.where(attn_mask[None, :], 0, -1e6).broadcast_to((N_SBLOCK, BLOCK_TOKENS))) * sm_scale # [BLOCK_TILE_Q * N_SBLOCK, BLOCK_TOKENS]

        max_new = tl.maximum(max_buffer, tl.max(qk, 1))
        qk = qk - max_new[:, None]
        p = tl.exp2(qk) # [BLOCK_TILE_Q * N_SBLOCK, BLOCK_TOKENS]
        denom_new = tl.sum(p, 1)
        alpha = tl.exp2(max_buffer - max_new)

        denom_buffer = denom_buffer * alpha + denom_new
        max_buffer = max_new
        output_buffer = output_buffer * alpha[:, None]
        v_values = tl.load(v_block_ptr, mask = attn_mask[:, None]).to(tl.float16)

        output_buffer = tl.dot(p.to(tl.float16), v_values, output_buffer) # [BLOCK_TILE_Q * N_SBLOCK, N_HEADDIM]

    max_buffer += tl.log2(denom_buffer)
    output_buffer = output_buffer / denom_buffer[:, None]

    tl.store(m_block_ptr, max_buffer)
    tl.store(o_block_ptr, output_buffer)

tiled_swa_bwd_preprocess_configs = [
    triton.Config({'BLOCK_Q': BQ }, num_stages=s, num_warps=w) \
    for BQ in [32, 64, 128]\
    for s in [3, 4, 7]\
    for w in [4, 8]\
]

@triton.autotune(tiled_swa_bwd_preprocess_configs, key=['N_TOKENS', 'N_HEADDIM'])
@triton.jit
def tiled_swa_bwd_preprocess_kernel(
    o, do, delta,
    o_stride0, o_stride1, o_stride2, o_stride3,
    do_stride0, do_stride1, do_stride2, do_stride3,
    m_stride0, m_stride1, m_stride2,
    N_TOKENS, 
    N_HEAD: tl.constexpr, N_HEADDIM: tl.constexpr,
    BLOCK_Q: tl.constexpr):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offset_batch = pid_bh // N_HEAD
    offset_head = pid_bh % N_HEAD

    o_bh_offsets = offset_batch * o_stride0 + offset_head * o_stride1
    do_bh_offsets = offset_batch * do_stride0 + offset_head * do_stride1
    delta_bh_offsets = offset_batch * m_stride0 + offset_head * m_stride1

    block_offsets = tl.arange(0, BLOCK_Q) + pid_q * BLOCK_Q
    head_dim_offsets = tl.arange(0, N_HEADDIM)

    block_mask = block_offsets < N_TOKENS

    do_values = tl.load(do + do_bh_offsets + block_offsets[:, None] * do_stride2 + head_dim_offsets[None, :] * do_stride3, mask = block_mask[:, None])
    o_values = tl.load(o + o_bh_offsets + block_offsets[:, None] * o_stride2 + head_dim_offsets[None, :] * o_stride3, mask = block_mask[:, None])

    delta_values = tl.sum(do_values * o_values, 1) # [BLOCK_Q,]
    tl.store(delta + delta_bh_offsets + block_offsets * m_stride2, delta_values, mask=block_mask)


tiled_swa_bwd_large_configs = [
    triton.Config({'BLOCK_TILE_Q': BQ }, num_stages=s, num_warps=w) \
    for BQ in [2, 4, 8]\
    for s in [3, 4, 7]\
    for w in [4, 8]\
]

# [B, H, N, N_KEYDIM/N_HEADDIM]
# m: [B, H, N]

@triton.autotune(tiled_swa_bwd_large_configs, key=['N_WIN_ROW', 'N_WIN_COL', 'N_SBLOCK', 'N_KEYDIM', 'N_HEADDIM'])
@triton.jit
def tiled_swa_bwd_large_kernel(
    q, k, v, m, do, delta,
    dq, dk, dv,
    q_stride0, q_stride1, q_stride2, q_stride3, 
    k_stride0, k_stride1, k_stride2, k_stride3, 
    v_stride0, v_stride1, v_stride2, v_stride3, 
    do_stride0, do_stride1, do_stride2, do_stride3,
    dqk_stride0, dqk_stride1, dqk_stride2, dqk_stride3, 
    dv_stride0, dv_stride1, dv_stride2, dv_stride3,
    m_stride0, m_stride1, m_stride2,
    sm_scale, N_TILES_ROW, N_TILES_COL, N_WIN_ROW: tl.constexpr, N_WIN_COL: tl.constexpr,
    N_HEAD: tl.constexpr, N_SBLOCK: tl.constexpr, N_KEYDIM: tl.constexpr, N_HEADDIM: tl.constexpr,
    BLOCK_TILE_Q: tl.constexpr):

    pid_kv = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    N_WINDOW_WIDTH: tl.constexpr = (2 * N_WIN_COL + 1)
    N_WINDOW_HEIGHT: tl.constexpr = (2 * N_WIN_ROW + 1)
    N_WINDOW_SIZE: tl.constexpr = N_WINDOW_WIDTH * N_WINDOW_HEIGHT

    offset_batch = pid_bh // N_HEAD
    offset_head = pid_bh % N_HEAD

    kv0_row = (pid_kv // N_TILES_COL)
    kv0_col = (pid_kv % N_TILES_COL)

    win0_row = kv0_row - N_WIN_ROW
    win0_col = kv0_col - N_WIN_COL
    
    q_bh_offsets = offset_batch * q_stride0 + offset_head * q_stride1
    k_bh_offsets = offset_batch * k_stride0 + offset_head * k_stride1
    v_bh_offsets = offset_batch * v_stride0 + offset_head * v_stride1
    do_bh_offsets = offset_batch * do_stride0 + offset_head * do_stride1
    m_bh_offsets = offset_batch * m_stride0 + offset_head * m_stride1
    dqk_bh_offsets = offset_batch * dqk_stride0 + offset_head * dqk_stride1
    dv_bh_offsets = offset_batch * dv_stride0 + offset_head * dv_stride1

    kv0_grid_offsets = kv0_row * N_TILES_COL + kv0_col

    sblock_offsets = tl.arange(0, N_SBLOCK)
    key_dim_offsets = tl.arange(0, N_KEYDIM)
    head_dim_offsets = tl.arange(0, N_HEADDIM)

    # [N_SBLOCK, N_KEYDIM]
    k_block_ptr = k + k_bh_offsets + (kv0_grid_offsets * N_SBLOCK + sblock_offsets[:, None]) * k_stride2 + key_dim_offsets[None, :] * k_stride3
    v_block_ptr = v + v_bh_offsets + (kv0_grid_offsets * N_SBLOCK + sblock_offsets[:, None]) * v_stride2 + head_dim_offsets[None, :] * v_stride3
    # [N_SBLOCK, N_HEADDIM]

    RCP_LN2: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471824645996

    k_values = (tl.load(k_block_ptr) * RCP_LN2 * sm_scale).to(tl.float16) # [N_SBLOCK, N_KEYDIM]
    v_values = tl.load(v_block_ptr).to(tl.float16) # [N_SBLOCK, N_HEADDIM]

    dk_buffer = tl.zeros(k_values.shape, dtype = tl.float32) # [N_SBLOCK, N_KEYDIM] 
    dv_buffer = tl.zeros(v_values.shape, dtype = tl.float32) # [N_SBLOCK, N_HEADDIM]
   
    BLOCK_TOKENS: tl.constexpr = N_SBLOCK * BLOCK_TILE_Q
    N_WINDOW_TOKENS: tl.constexpr = N_SBLOCK * N_WINDOW_SIZE

    for idx in range(0, N_WINDOW_TOKENS, BLOCK_TOKENS):
        q_idx = idx + tl.arange(0, BLOCK_TOKENS)

        tile_idx = q_idx // N_SBLOCK
        token_idx = q_idx % N_SBLOCK

        tile_row = win0_row + (tile_idx // N_WINDOW_WIDTH) # [BLOCK_TOKENS, ]
        tile_col = win0_col + (tile_idx % N_WINDOW_WIDTH) # [BLOCK_TOKENS, ]
        
        attn_mask = ((tile_row >= 0) & (tile_row < N_TILES_ROW) & 
                    (tile_col >= 0) & (tile_col < N_TILES_COL) & 
                    (q_idx < N_WINDOW_TOKENS)) # [BLOCK_TOKENS, ]
    
        oq_token_idx = ((tile_row * N_TILES_COL + tile_col) * N_SBLOCK + token_idx)

        m_offsets = m_bh_offsets + oq_token_idx * m_stride2 # [BLOCK_TOKENS, ]
        q_offsets = q_bh_offsets + oq_token_idx * q_stride2 # [BLOCK_TOKENS, ]
        do_offsets = do_bh_offsets + oq_token_idx * do_stride2 # [BLOCK_TOKENS, ]
        delta_ptrs = delta + m_bh_offsets + oq_token_idx * m_stride2
        
        q_block_ptr = q + q_offsets[None, :] + key_dim_offsets[:, None] * q_stride3 # [N_KEYDIM, BLOCK_TOKENS]
        m_values = tl.load(m + m_offsets, mask = attn_mask) # [BLOCK_TOKENS, ]
        q_values = tl.load(q_block_ptr, mask = attn_mask[None, :]).to(tl.float16) # [N_KEYDIM, BLOCK_TOKENS]

        qk_init = tl.where(attn_mask[None, :], 0, -1e6).broadcast_to((N_SBLOCK, BLOCK_TOKENS))
        qkT = tl.dot(k_values, q_values.to(tl.float16), qk_init) # [N_SBLOCK, BLOCK_TOKENS]
        pT = tl.math.exp2(qkT - m_values[None, :]) # [N_SBLOCK, BLOCK_TOKENS]

        do_values = tl.load(do + do_offsets[:, None] + head_dim_offsets[None,:] * do_stride3, mask = attn_mask[:, None]).to(tl.float16) # [BLOCK_TOKENS, N_HEADDIM]

        dv_buffer += tl.dot(pT.to(tl.float16), do_values)  # [N_SBLOCK, N_HEADDIM]
        # D (= delta) is pre-divided by ds_scale.
        delta_values = tl.load(delta_ptrs, mask = attn_mask) # [BLOCK_TOKENS, ]
        # Compute dP and dS.
        dpT = tl.dot(v_values, tl.trans(do_values)).to(tl.float32) # [N_SBLOCK, BLOCK_TOKENS]
        dsT = (pT * (dpT - delta_values[None, :])).to(tl.float16) # [N_SBLOCK, BLOCK_TOKENS]
        dk_buffer += tl.dot(dsT, tl.trans(q_values))

        dq_values = tl.dot(tl.trans(dsT), (k_values)) * LN2 # [BLOCK_TOKENS, N_KEYDIM]

        dq_ptrs = dq + dqk_bh_offsets + oq_token_idx[:, None] * dqk_stride2 + key_dim_offsets[None, :] * dqk_stride3
        tl.atomic_add(dq_ptrs, dq_values, mask = attn_mask[:, None])

    dk_ptrs = dk + dqk_bh_offsets + (kv0_grid_offsets * N_SBLOCK + sblock_offsets[:, None]) * dqk_stride2 + key_dim_offsets[None, :] * dqk_stride3
    tl.store(dk_ptrs, dk_buffer * sm_scale)

    dv_ptrs = dv + dv_bh_offsets + (kv0_grid_offsets * N_SBLOCK + sblock_offsets[:, None]) * dv_stride2 + head_dim_offsets[None, :] * dv_stride3
    tl.store(dv_ptrs, dv_buffer)

@triton.autotune(tiled_swa_bwd_large_configs, key=['N_WIN_ROW', 'N_WIN_COL', 'N_SBLOCK', 'N_KEYDIM', 'N_HEADDIM'])
@triton.jit
def tiled_swa_bwd_large_kernel_decoupled_q(
    q, k, v, m, do, delta,
    dq, dk, dv,
    q_stride0, q_stride1, q_stride2, q_stride3, 
    k_stride0, k_stride1, k_stride2, k_stride3, 
    v_stride0, v_stride1, v_stride2, v_stride3, 
    do_stride0, do_stride1, do_stride2, do_stride3,
    dqk_stride0, dqk_stride1, dqk_stride2, dqk_stride3, 
    dv_stride0, dv_stride1, dv_stride2, dv_stride3,
    m_stride0, m_stride1, m_stride2,
    sm_scale, N_TILES_ROW, N_TILES_COL, N_WIN_ROW: tl.constexpr, N_WIN_COL: tl.constexpr,
    N_HEAD: tl.constexpr, N_SBLOCK: tl.constexpr, N_KEYDIM: tl.constexpr, N_HEADDIM: tl.constexpr,
    BLOCK_TILE_Q: tl.constexpr):

    pid_kv = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    N_WINDOW_WIDTH: tl.constexpr = (2 * N_WIN_COL + 1)
    N_WINDOW_HEIGHT: tl.constexpr = (2 * N_WIN_ROW + 1)
    N_WINDOW_SIZE: tl.constexpr = N_WINDOW_WIDTH * N_WINDOW_HEIGHT

    offset_batch = pid_bh // N_HEAD
    offset_head = pid_bh % N_HEAD

    kv0_row = (pid_kv // N_TILES_COL)
    kv0_col = (pid_kv % N_TILES_COL)

    win0_row = kv0_row - N_WIN_ROW
    win0_col = kv0_col - N_WIN_COL
    
    q_bh_offsets = offset_batch * q_stride0 + offset_head * q_stride1
    k_bh_offsets = offset_batch * k_stride0 + offset_head * k_stride1
    v_bh_offsets = offset_batch * v_stride0 + offset_head * v_stride1
    do_bh_offsets = offset_batch * do_stride0 + offset_head * do_stride1
    m_bh_offsets = offset_batch * m_stride0 + offset_head * m_stride1
    dqk_bh_offsets = offset_batch * dqk_stride0 + offset_head * dqk_stride1
    dv_bh_offsets = offset_batch * dv_stride0 + offset_head * dv_stride1

    kv0_grid_offsets = kv0_row * N_TILES_COL + kv0_col

    sblock_offsets = tl.arange(0, N_SBLOCK)
    key_dim_offsets = tl.arange(0, N_KEYDIM)
    head_dim_offsets = tl.arange(0, N_HEADDIM)

    # [N_SBLOCK, N_KEYDIM]
    k_block_ptr = k + k_bh_offsets + (kv0_grid_offsets * N_SBLOCK + sblock_offsets[:, None]) * k_stride2 + key_dim_offsets[None, :] * k_stride3
    v_block_ptr = v + v_bh_offsets + (kv0_grid_offsets * N_SBLOCK + sblock_offsets[:, None]) * v_stride2 + head_dim_offsets[None, :] * v_stride3
    self_q_block_ptr = q + q_bh_offsets + (kv0_grid_offsets * N_SBLOCK + sblock_offsets[None, :]) * q_stride2 + key_dim_offsets[:, None] * q_stride3
    self_m_block_ptr = m + m_bh_offsets + (kv0_grid_offsets * N_SBLOCK + sblock_offsets) * m_stride2
    self_delta_block_ptr = delta + m_bh_offsets + (kv0_grid_offsets * N_SBLOCK + sblock_offsets) * m_stride2
    self_do_block_ptr = do + do_bh_offsets + (kv0_grid_offsets * N_SBLOCK + sblock_offsets[:, None]) * do_stride2 + head_dim_offsets[None, :] * do_stride3
    self_dq_block_ptr = dq + dqk_bh_offsets + (kv0_grid_offsets * N_SBLOCK + sblock_offsets[:,None]) * dqk_stride2 + key_dim_offsets[None, :] * dqk_stride3
    # [N_SBLOCK, N_HEADDIM]

    RCP_LN2: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471824645996

    k_values = (tl.load(k_block_ptr) * RCP_LN2 * sm_scale).to(tl.float16) # [N_SBLOCK, N_KEYDIM]
    v_values = tl.load(v_block_ptr).to(tl.float16) # [N_SBLOCK, N_HEADDIM]

    dk_buffer = tl.zeros(k_values.shape, dtype = tl.float32) # [N_SBLOCK, N_KEYDIM] 
    dv_buffer = tl.zeros(v_values.shape, dtype = tl.float32) # [N_SBLOCK, N_HEADDIM]
   
    BLOCK_TOKENS: tl.constexpr = N_SBLOCK * BLOCK_TILE_Q
    N_WINDOW_TOKENS: tl.constexpr = N_SBLOCK * N_WINDOW_SIZE

    for idx in range(0, N_WINDOW_TOKENS, BLOCK_TOKENS):
        q_idx = idx + tl.arange(0, BLOCK_TOKENS)

        tile_idx = q_idx // N_SBLOCK
        token_idx = q_idx % N_SBLOCK

        tile_row = win0_row + (tile_idx // N_WINDOW_WIDTH) # [BLOCK_TOKENS, ]
        tile_col = win0_col + (tile_idx % N_WINDOW_WIDTH) # [BLOCK_TOKENS, ]
        
        attn_mask = ((tile_row >= 0) & (tile_row < N_TILES_ROW) & 
                    (tile_col >= 0) & (tile_col < N_TILES_COL) & 
                    (q_idx < N_WINDOW_TOKENS)) # [BLOCK_TOKENS, ]
    
        oq_token_idx = ((tile_row * N_TILES_COL + tile_col) * N_SBLOCK + token_idx)

        m_offsets = m_bh_offsets + oq_token_idx * m_stride2 # [BLOCK_TOKENS, ]
        q_offsets = q_bh_offsets + oq_token_idx * q_stride2 # [BLOCK_TOKENS, ]
        do_offsets = do_bh_offsets + oq_token_idx * do_stride2 # [BLOCK_TOKENS, ]
        delta_ptrs = delta + m_bh_offsets + oq_token_idx * m_stride2
        
        q_block_ptr = q + q_offsets[None, :] + key_dim_offsets[:, None] * q_stride3 # [N_KEYDIM, BLOCK_TOKENS]
        m_values = tl.load(m + m_offsets, mask = attn_mask) # [BLOCK_TOKENS, ]
        q_values = tl.load(q_block_ptr, mask = attn_mask[None, :]).to(tl.float16) # [N_KEYDIM, BLOCK_TOKENS]

        qk_init = tl.where(attn_mask[None, :], 0, -1e6).broadcast_to((N_SBLOCK, BLOCK_TOKENS))
        qkT = tl.dot(k_values, q_values.to(tl.float16), qk_init) # [N_SBLOCK, BLOCK_TOKENS]
        pT = tl.math.exp2(qkT - m_values[None, :]) # [N_SBLOCK, BLOCK_TOKENS]

        do_values = tl.load(do + do_offsets[:, None] + head_dim_offsets[None,:] * do_stride3, mask = attn_mask[:, None]).to(tl.float16) # [BLOCK_TOKENS, N_HEADDIM]

        dv_buffer += tl.dot(pT.to(tl.float16), do_values)  # [N_SBLOCK, N_HEADDIM]
        # D (= delta) is pre-divided by ds_scale.
        delta_values = tl.load(delta_ptrs, mask = attn_mask) # [BLOCK_TOKENS, ]
        # Compute dP and dS.
        dpT = tl.dot(v_values, tl.trans(do_values)).to(tl.float32) # [N_SBLOCK, BLOCK_TOKENS]
        dsT = (pT * (dpT - delta_values[None, :])).to(tl.float16) # [N_SBLOCK, BLOCK_TOKENS]
        dk_buffer += tl.dot(dsT, tl.trans(q_values))


    dk_ptrs = dk + dqk_bh_offsets + (kv0_grid_offsets * N_SBLOCK + sblock_offsets[:, None]) * dqk_stride2 + key_dim_offsets[None, :] * dqk_stride3
    tl.store(dk_ptrs, dk_buffer * sm_scale)

    dv_ptrs = dv + dv_bh_offsets + (kv0_grid_offsets * N_SBLOCK + sblock_offsets[:, None]) * dv_stride2 + head_dim_offsets[None, :] * dv_stride3
    tl.store(dv_ptrs, dv_buffer)
    
    this_m_values = tl.load(self_m_block_ptr)
    this_q_values = (tl.load(self_q_block_ptr) * sm_scale * RCP_LN2).to(tl.float16) # [N_KEYDIM, N_SBLOCK]
    this_do_values = tl.load(self_do_block_ptr).to(tl.float16) # [N_SBLOCK, N_HEADDIM]
    this_delta_values = tl.load(self_delta_block_ptr).to(tl.float16) # [N_SBLOCK, ]

    dq_buffer = tl.zeros((N_SBLOCK, N_KEYDIM), dtype = tl.float32)

    for idx in range(0, N_WINDOW_TOKENS, BLOCK_TOKENS):
        kv_idx = idx + tl.arange(0, BLOCK_TOKENS)

        tile_idx = kv_idx // N_SBLOCK
        token_idx = kv_idx % N_SBLOCK

        tile_row = win0_row + (tile_idx // N_WINDOW_WIDTH) # [BLOCK_TOKENS, ]
        tile_col = win0_col + (tile_idx % N_WINDOW_WIDTH) # [BLOCK_TOKENS, ]
        
        target_attn_mask = ((tile_row >= 0) & (tile_row < N_TILES_ROW) & 
                    (tile_col >= 0) & (tile_col < N_TILES_COL) & 
                    (kv_idx < N_WINDOW_TOKENS)) # [BLOCK_TOKENS, ]
    
        kv_token_idx = ((tile_row * N_TILES_COL + tile_col) * N_SBLOCK + token_idx) # [BLOCK_TOKENS, ]

        target_k_offsets = k_bh_offsets + kv_token_idx * k_stride2 # [BLOCK_TOKENS, ]
        target_v_offsets = v_bh_offsets + kv_token_idx * v_stride2
        
        target_k_block_ptr = k + target_k_offsets[:,None] + key_dim_offsets[None, :] * k_stride3 # [BLOCK_TOKENS, N_KEYDIM]
        target_v_block_ptr = v + target_v_offsets[:, None] + head_dim_offsets[None, :] * v_stride3 # [BLOCK_TOKENS, N_HEADDIM]
        
        target_k_values = tl.load(target_k_block_ptr, mask = target_attn_mask[:, None]) # [BLOCK_TOKENS, N_KEYDIM]
        
        target_qk_init = tl.where(target_attn_mask[:, None], 0, -1e6).broadcast_to((BLOCK_TOKENS, N_SBLOCK))
        target_qkT = tl.dot(target_k_values, this_q_values, target_qk_init) # [BLOCK_TOKENS, N_SBLOCK]
        target_pT = tl.math.exp2(target_qkT - this_m_values[None, :]) # [BLOCK_TOKENS, N_SBLOCK]

        target_v_values = tl.load(target_v_block_ptr, mask = target_attn_mask[:, None]) # [BLOCK_TOKENS, N_HEADDIM]
        # Compute dP and dS.
        # [BLOCK_TOKENS, N_HEADDIM] x [N_SBLOCK, N_HEADDIM]^T
        target_dpT = tl.dot(target_v_values, tl.trans(this_do_values)).to(tl.float32) # [BLOCK_TOKENS, N_SBLOCK]
        dsT = (target_pT * (target_dpT - this_delta_values[None, :])).to(tl.float16) # [BLOCK_TOKENS, N_SBLOCK]

        dq_buffer = tl.dot(tl.trans(dsT), (target_k_values), dq_buffer) # [N_SBLOCK, N_KEYDIM]

    tl.store(self_dq_block_ptr, dq_buffer * sm_scale)

def tiled_swa_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b_size: int, t_height: int, t_width: int, row_size: int, col_size: int, sm_scale: float = 1.0):
    B, H, N, KEY_DIM = q.shape
    HEAD_DIM = v.shape[-1]

    m = torch.empty((B, H, N), dtype = torch.float16)
    o = torch.empty((B, H, N, HEAD_DIM), dtype = torch.float16)

    if b_size < 16:
        grid = lambda meta: (triton.cdiv(t_width, meta['BLOCK_TILE_Q_COL']) * triton.cdiv(t_height, meta['BLOCK_TILE_Q_ROW']), B * H)
        tiled_swa_fwd_kernel[grid](
            q, k, v, m, o,
            q.stride(0),q.stride(1),q.stride(2),q.stride(3),
            k.stride(0),k.stride(1),k.stride(2),k.stride(3),
            v.stride(0),v.stride(1),v.stride(2),v.stride(3),
            o.stride(0),o.stride(1),o.stride(2),o.stride(3),
            m.stride(0),m.stride(1),m.stride(2),
            sm_scale, t_height, t_width,
            row_size, col_size, 
            H, b_size, KEY_DIM, HEAD_DIM, 
            BLOCK_TILE_KV=4, BLOCK_TILE_Q_COL=2, BLOCK_TILE_Q_ROW=2
        )
    else:
        grid = lambda meta: (t_width * t_height, B * H)
        tiled_swa_fwd_large_kernel[grid](
            q, k, v, m, o,
            q.stride(0),q.stride(1),q.stride(2),q.stride(3),
            k.stride(0),k.stride(1),k.stride(2),k.stride(3),
            v.stride(0),v.stride(1),v.stride(2),v.stride(3),
            o.stride(0),o.stride(1),o.stride(2),o.stride(3),
            m.stride(0),m.stride(1),m.stride(2),
            sm_scale, t_height, t_width,
            row_size, col_size, 
            H, b_size, KEY_DIM, HEAD_DIM, 
            BLOCK_TILE_KV=4
        )

    return o

class TiledSwa(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.Function, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b_size: int, t_height: int, t_width: int, row_size: int, col_size: int, sm_scale: float):
        B, H, N, KEY_DIM = q.shape
        HEAD_DIM = v.shape[-1]

        m = torch.empty((B, H, N), dtype = torch.float16)
        o = torch.empty((B, H, N, HEAD_DIM), dtype = torch.float16)

        if b_size < 16:
            grid = lambda meta: (triton.cdiv(t_width, meta['BLOCK_TILE_Q_COL']) * triton.cdiv(t_height, meta['BLOCK_TILE_Q_ROW']), B * H)
            tiled_swa_fwd_kernel[grid](
                q, k, v, m, o,
                q.stride(0),q.stride(1),q.stride(2),q.stride(3),
                k.stride(0),k.stride(1),k.stride(2),k.stride(3),
                v.stride(0),v.stride(1),v.stride(2),v.stride(3),
                o.stride(0),o.stride(1),o.stride(2),o.stride(3),
                m.stride(0),m.stride(1),m.stride(2),
                sm_scale, t_height, t_width,
                row_size, col_size, 
                H, b_size, KEY_DIM, HEAD_DIM, 
                BLOCK_TILE_KV=4, BLOCK_TILE_Q_COL=2, BLOCK_TILE_Q_ROW=2
            )
        else:
            grid = lambda meta: (t_width * t_height, B * H)
            tiled_swa_fwd_large_kernel[grid](
                q, k, v, m, o,
                q.stride(0),q.stride(1),q.stride(2),q.stride(3),
                k.stride(0),k.stride(1),k.stride(2),k.stride(3),
                v.stride(0),v.stride(1),v.stride(2),v.stride(3),
                o.stride(0),o.stride(1),o.stride(2),o.stride(3),
                m.stride(0),m.stride(1),m.stride(2),
                sm_scale, t_height, t_width,
                row_size, col_size, 
                H, b_size, KEY_DIM, HEAD_DIM
            )

        ctx.save_for_backward(q, k, v, m, o)
        ctx.b_size = b_size
        ctx.t_height = t_height
        ctx.t_width = t_width
        ctx.row_size = row_size
        ctx.col_size = col_size
        ctx.sm_scale = sm_scale
        return o
    
    @staticmethod
    def backward(ctx: torch.autograd.Function, do: torch.Tensor):
        q, k, v, m, o = ctx.saved_tensors
        b_size, t_height, t_width, row_size, col_size, sm_scale = ctx.b_size, ctx.t_height, ctx.t_width, ctx.row_size, ctx.col_size, ctx.sm_scale

        B, H, N, KEY_DIM = q.shape
        HEAD_DIM = v.shape[-1]

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        if b_size < 16:
            raise NotImplementedError("Tile size < 16 not supported")
        else:
            delta = torch.empty((B, H, N), dtype = torch.float32, device = o.device)
            pre_grid = lambda meta: (triton.cdiv(N, meta['BLOCK_Q']) ,B * H)

            tiled_swa_bwd_preprocess_kernel[pre_grid](
                o, do, delta, 
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                delta.stride(0), delta.stride(1), delta.stride(2),
                N, H, HEAD_DIM
            )

            grid = lambda meta: (t_width * t_height, B * H)
            tiled_swa_bwd_large_kernel_decoupled_q[grid](
                q, k, v, m, do, delta, 
                dq, dk, dv, 
                q.stride(0),q.stride(1),q.stride(2),q.stride(3),
                k.stride(0),k.stride(1),k.stride(2),k.stride(3),
                v.stride(0),v.stride(1),v.stride(2),v.stride(3),
                do.stride(0),do.stride(1),do.stride(2),do.stride(3),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                m.stride(0), m.stride(1), m.stride(2),
                sm_scale, t_height, t_width,
                row_size, col_size, H, b_size, KEY_DIM, HEAD_DIM
            )
        
        return dq, dk, dv, None, None, None, None, None, None, 
    

# [B, H, N, S, KEY_DIM/HEAD_DIM]
def tiled_swa_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b_size: int, t_height: int, t_width: int, row_size: int, col_size: int, sm_scale: float = 1.0) -> torch.Tensor:
    B, H, N, KEY_DIM = q.shape
    HEAD_DIM = v.shape[-1]

    n_tiles = t_height * t_width

    assert N == n_tiles * b_size, "seqlen should match image size"

    row_indices = torch.arange(0, t_height) # [h]
    col_indices = torch.arange(0, t_width) # [w]

    row_mask = torch.logical_and(
        row_indices[None, :] >= row_indices[:, None] - row_size,  
        row_indices[None, :] <= row_indices[:, None] + row_size)
    
    col_mask = torch.logical_and(
        col_indices[None, :] >= col_indices[:, None] - col_size,  
        col_indices[None, :] <= col_indices[:, None] + col_size)
    
    total_mask = torch.where(torch.logical_and(row_mask[:, None, :, None], col_mask[None, :, None, :]), 0, -1e6).to(q.dtype)
    attn_mask = total_mask.reshape((n_tiles, 1, n_tiles, 1)).expand((n_tiles, b_size, n_tiles, b_size)).reshape((N, N))

    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=sm_scale)

    return o



def run_backward(o: torch.Tensor):
    if not torch.is_grad_enabled():
        return o
    if not (o.shape in shape_cache):
        print(f"Create grad for {o.shape}")
        shape_cache[o.shape] = torch.randn_like(o)
    do = shape_cache[o.shape]
    o.backward(do)

if __name__ == "__main__":
    shape_cache = dict()

    torch.set_default_device('cuda:0')
    torch.manual_seed(0)

    B, H, KEY_DIM, HEAD_DIM = 4, 8, 32, 32
    use_fused_na()
    use_autotuner()
    use_kv_parallelism_in_fused_na(True)

    BH, BW = 4, 4
    BLOCK_SIZE = BH * BW
    TILE_WIDTH, TILE_HEIGHT = 16, 16
    N = BLOCK_SIZE * TILE_WIDTH * TILE_HEIGHT

    ROW_SIZE, COL_SIZE = 1,1

    q = torch.randn((B, H, N, KEY_DIM) ).to(torch.float16)
    k = torch.randn((B, H, N, KEY_DIM) ).to(torch.float16)
    v = torch.randn((B, H, N, HEAD_DIM)).to(torch.float16)

    
    q_o = q.clone().reshape((B, H, TILE_HEIGHT, TILE_WIDTH, BH, BW, KEY_DIM)).permute((0, 2, 4, 3, 5, 1, 6)).reshape((B, TILE_HEIGHT * BH, TILE_WIDTH * BW, H, KEY_DIM)).requires_grad_(True)
    k_o = k.clone().reshape((B, H, TILE_HEIGHT, TILE_WIDTH, BH, BW, KEY_DIM)).permute((0, 2, 4, 3, 5, 1, 6)).reshape((B, TILE_HEIGHT * BH, TILE_WIDTH * BW, H, KEY_DIM)).requires_grad_(True)
    v_o = v.clone().reshape((B, H, TILE_HEIGHT, TILE_WIDTH, BH, BW, HEAD_DIM)).permute((0, 2, 4, 3, 5, 1, 6)).reshape((B, TILE_HEIGHT * BH, TILE_WIDTH * BW, H, HEAD_DIM)).requires_grad_(True)

    
    q0, k0, v0 = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
    q1, k1, v1 = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)

    flops_per_matmul = 2.0 * B * H * N * N * HEAD_DIM
    total_tflops_dms = 2 * flops_per_matmul * 1e-9

    with torch.set_grad_enabled(True):
        m0 = total_tflops_dms / do_bench(lambda: run_backward(na2d(q_o, k_o, v_o, kernel_size=11)))
        m1 = total_tflops_dms / do_bench(lambda: run_backward(tiled_swa_ref(q0, k0, v0, BLOCK_SIZE, TILE_HEIGHT, TILE_WIDTH, ROW_SIZE, COL_SIZE, 0.25)))
        m2 = total_tflops_dms / do_bench(lambda: run_backward(TiledSwa.apply(q0, k0, v0, BLOCK_SIZE, TILE_HEIGHT, TILE_WIDTH, ROW_SIZE, COL_SIZE, 0.25)))

    print(m0, m1, m2)

    with torch.set_grad_enabled(False):
        m0 = total_tflops_dms / do_bench(lambda: run_backward(na2d(q_o, k_o, v_o, kernel_size=19)))
        m1 = total_tflops_dms / do_bench(lambda: run_backward(tiled_swa_ref(q0, k0, v0, BLOCK_SIZE, TILE_HEIGHT, TILE_WIDTH, ROW_SIZE, COL_SIZE, 0.25)))
        m2 = total_tflops_dms / do_bench(lambda: run_backward(TiledSwa.apply(q0, k0, v0, BLOCK_SIZE, TILE_HEIGHT, TILE_WIDTH, ROW_SIZE, COL_SIZE, 0.25)))

    print(m0, m1, m2)
    exit(0)

    o = tiled_swa_ref(q0, k0, v0, BLOCK_SIZE, TILE_HEIGHT, TILE_WIDTH, ROW_SIZE, COL_SIZE, 0.25)
    o2 = TiledSwa.apply(q1, k1, v1, BLOCK_SIZE, TILE_HEIGHT, TILE_WIDTH, ROW_SIZE, COL_SIZE, 0.25)

    o.flatten().sum().backward()
    o2.flatten().sum().backward()

    print(torch.abs(q0.grad - q1.grad).max())
    print(torch.abs(k0.grad - k1.grad).max())
    print(torch.abs(v0.grad - v1.grad).max())

    pass