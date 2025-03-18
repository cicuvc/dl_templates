import os
# os.environ['TRITON_INTERPRET']='1'

import triton
import triton.language as tl
import torch

from .rope2d import FusedRotaryPositionEncodingContext


configs = [
    triton.Config({'BLOCK_BATCH': BB, 'BLOCK_TOKEN': BT}, num_stages=s, num_warps=w) \
    for BB in [8, 16]\
    for BT in [16, 32, 64]\
    for s in [4, 7]\
    for w in [4, 8]\
]


# x: [B, N, D]
# idx: [N, ]
# row_cs_buffer: [N_MAX, N_RDIM, 2]
# col_cs_buffer: [N_MAX, N_RDIM, 2]
@triton.autotune(configs, key=["N_TOTAL", "N_TOKENS"])
@triton.jit
def rope2d_fwd_kernel(
    x, idx, row_cs_buffer, col_cs_buffer, 
    x_stride0, x_stride1, x_stride2,
    cs_stride0,

    N_TOTAL, N_TOKENS, N_RDIM: tl.constexpr,
    BLOCK_BATCH: tl.constexpr, BLOCK_TOKEN: tl.constexpr):

    pid_block = tl.program_id(0) # merge block to promote data reuse from cs_buffers
    pid_batch = tl.program_id(1)

    block_offsets = tl.arange(0, BLOCK_TOKEN) + pid_block * BLOCK_TOKEN
    batch_offsets = tl.arange(0, BLOCK_BATCH) + pid_batch * BLOCK_BATCH

    idx_values = tl.load(idx + block_offsets, mask = block_offsets < N_TOKENS, other=-1) # [BLOCK_TOKEN]
    row_values = (idx_values >> 16).to(tl.int16) # [BLOCK_TOKEN]
    col_values = (idx_values & 0xFFFF).to(tl.int16) # [BLOCK_TOKEN]
    
    row_mask = row_values[:, None, None] >= 0
    col_mask = col_values[:, None, None] >= 0

    # early exit to save memory bandwidth
    if not tl.max(tl.maximum(row_mask, col_mask)).to(tl.int1):
        return
    
    cs_offsets = tl.arange(0, N_RDIM)[None, :, None] * 2 + tl.arange(0, 2)[None, None, :]

    row_cs = tl.load(row_cs_buffer + row_values[:, None, None] * cs_stride0 + cs_offsets, mask = row_mask, other=1) # [BLOCK_TOKEN, 2]
    col_cs = tl.load(col_cs_buffer + col_values[:, None, None] * cs_stride0 + cs_offsets, mask = col_mask, other=0) # [BLOCK_TOKEN, 2]

    row_cos, row_sin = tl.split(row_cs)
    col_cos, col_sin = tl.split(col_cs) # [BLOCK_TOKEN, N_RDIM]

    
    xr0_offsets = batch_offsets[:, None, None] * x_stride0 + block_offsets[None, :, None] * x_stride1 + tl.arange(0, N_RDIM) * x_stride2
    xr1_offsets = batch_offsets[:, None, None] * x_stride0 + block_offsets[None, :, None] * x_stride1 + tl.arange(N_RDIM, N_RDIM * 2) * x_stride2
    xr0_values = tl.load(x + xr0_offsets, mask = xr0_offsets < N_TOTAL) # [BLOCK_BATCH, BLOCK_TOKEN, N_RDIM]
    xr1_values = tl.load(x + xr1_offsets, mask = xr1_offsets < N_TOTAL) # [BLOCK_BATCH, BLOCK_TOKEN, N_RDIM]
    tl.store(x + xr0_offsets, xr0_values * row_cos[None, :, :] - xr1_values * row_sin[None, :, :], mask = xr0_offsets < N_TOTAL)
    tl.store(x + xr1_offsets, xr0_values * row_sin[None, :, :] + xr1_values * row_cos[None, :, :], mask = xr1_offsets < N_TOTAL)

    xc0_offsets = batch_offsets[:, None, None] * x_stride0 + block_offsets[None, :, None] * x_stride1 + tl.arange(N_RDIM * 2, N_RDIM * 3) * x_stride2
    xc1_offsets = batch_offsets[:, None, None] * x_stride0 + block_offsets[None, :, None] * x_stride1 + tl.arange(N_RDIM * 3, N_RDIM * 4) * x_stride2
    xc0_values = tl.load(x + xc0_offsets, mask = xc0_offsets < N_TOTAL) # [BLOCK_BATCH, BLOCK_TOKEN, N_RDIM]
    xc1_values = tl.load(x + xc1_offsets, mask = xc1_offsets < N_TOTAL) # [BLOCK_BATCH, BLOCK_TOKEN, N_RDIM]
    tl.store(x + xc0_offsets, xc0_values * col_cos[None, :, :] - xc1_values * col_sin[None, :, :], mask = xc0_offsets < N_TOTAL)
    tl.store(x + xc1_offsets, xc0_values * col_sin[None, :, :] + xc1_values * col_cos[None, :, :], mask = xc1_offsets < N_TOTAL)

@triton.autotune(configs, key=["N_TOTAL", "N_TOKENS"])
@triton.jit
def rope2d_bwd_kernel(
    x, idx, row_cs_buffer, col_cs_buffer, 
    x_stride0, x_stride1, x_stride2,
    cs_stride0,

    N_TOTAL, N_TOKENS, N_RDIM: tl.constexpr,
    BLOCK_BATCH: tl.constexpr, BLOCK_TOKEN: tl.constexpr):

    pid_block = tl.program_id(0) # merge block to promote data reuse from cs_buffers
    pid_batch = tl.program_id(1)

    block_offsets = tl.arange(0, BLOCK_TOKEN) + pid_block * BLOCK_TOKEN
    batch_offsets = tl.arange(0, BLOCK_BATCH) + pid_batch * BLOCK_BATCH

    idx_values = tl.load(idx + block_offsets, mask = block_offsets < N_TOKENS, other=-1) # [BLOCK_TOKEN]
    row_values = (idx_values >> 16).to(tl.int16) # [BLOCK_TOKEN]
    col_values = (idx_values & 0xFFFF).to(tl.int16) # [BLOCK_TOKEN]
    
    row_mask = row_values[:, None, None] >= 0
    col_mask = col_values[:, None, None] >= 0

    # early exit to save memory bandwidth
    if not tl.max(tl.maximum(row_mask, col_mask)).to(tl.int1):
        return
    
    cs_offsets = tl.arange(0, N_RDIM)[None, :, None] * 2 + tl.arange(0, 2)[None, None, :]

    row_cs = tl.load(row_cs_buffer + row_values[:, None, None] * cs_stride0 + cs_offsets, mask = row_mask, other=1) # [BLOCK_TOKEN, 2]
    col_cs = tl.load(col_cs_buffer + col_values[:, None, None] * cs_stride0 + cs_offsets, mask = col_mask, other=0) # [BLOCK_TOKEN, 2]

    row_cos, row_sin = tl.split(row_cs)
    col_cos, col_sin = tl.split(col_cs) # [BLOCK_TOKEN, N_RDIM]

    
    xr0_offsets = batch_offsets[:, None, None] * x_stride0 + block_offsets[None, :, None] * x_stride1 + tl.arange(0, N_RDIM) * x_stride2
    xr1_offsets = batch_offsets[:, None, None] * x_stride0 + block_offsets[None, :, None] * x_stride1 + tl.arange(N_RDIM, N_RDIM * 2) * x_stride2
    xr0_values = tl.load(x + xr0_offsets, mask = xr0_offsets < N_TOTAL) # [BLOCK_BATCH, BLOCK_TOKEN, N_RDIM]
    xr1_values = tl.load(x + xr1_offsets, mask = xr1_offsets < N_TOTAL) # [BLOCK_BATCH, BLOCK_TOKEN, N_RDIM]
    tl.store(x + xr0_offsets, xr0_values * row_cos[None, :, :] + xr1_values * row_sin[None, :, :], mask = xr0_offsets < N_TOTAL)
    tl.store(x + xr1_offsets, -xr0_values * row_sin[None, :, :] + xr1_values * row_cos[None, :, :], mask = xr1_offsets < N_TOTAL)

    xc0_offsets = batch_offsets[:, None, None] * x_stride0 + block_offsets[None, :, None] * x_stride1 + tl.arange(N_RDIM * 2, N_RDIM * 3) * x_stride2
    xc1_offsets = batch_offsets[:, None, None] * x_stride0 + block_offsets[None, :, None] * x_stride1 + tl.arange(N_RDIM * 3, N_RDIM * 4) * x_stride2
    xc0_values = tl.load(x + xc0_offsets, mask = xc0_offsets < N_TOTAL) # [BLOCK_BATCH, BLOCK_TOKEN, N_RDIM]
    xc1_values = tl.load(x + xc1_offsets, mask = xc1_offsets < N_TOTAL) # [BLOCK_BATCH, BLOCK_TOKEN, N_RDIM]
    tl.store(x + xc0_offsets, xc0_values * col_cos[None, :, :] + xc1_values * col_sin[None, :, :], mask = xc0_offsets < N_TOTAL)
    tl.store(x + xc1_offsets, -xc0_values * col_sin[None, :, :] + xc1_values * col_cos[None, :, :], mask = xc1_offsets < N_TOTAL)

class RotaryPE2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function._ContextMethodMixin, x: torch.Tensor, idx: torch.Tensor, row_cs_buffer: torch.Tensor, col_cs_buffer: torch.Tensor):
        B, N, N_DIM = x.shape
        N_MAX_ROW, N_RDIM, _ = row_cs_buffer.shape

        assert row_cs_buffer.shape[1] == col_cs_buffer.shape[1]
        assert idx.shape[0] == N

        N_TOTAL = x.untyped_storage().size() // x.element_size()

        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_TOKEN']), triton.cdiv(B, meta['BLOCK_BATCH']))

        torch.cuda.set_device(x.device)
        rope2d_fwd_kernel[grid](
            x, idx, row_cs_buffer, col_cs_buffer, 
            x.stride(0), x.stride(1), x.stride(2), 
            row_cs_buffer.stride(0), N_TOTAL, N, N_RDIM)

        ctx.mark_dirty(x)
        ctx.save_for_backward(x, idx, row_cs_buffer, col_cs_buffer)
        return x

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: torch.autograd.function._ContextMethodMixin, do: torch.Tensor):  
        x, idx, row_cs_buffer, col_cs_buffer = ctx.saved_tensors
        dx = do.clone()

        B, N, N_DIM = dx.shape
        N_MAX_ROW, N_RDIM, _ = row_cs_buffer.shape

        assert row_cs_buffer.shape[1] == col_cs_buffer.shape[1]
        assert idx.shape[0] == N

        N_TOTAL = x.untyped_storage().size() // x.element_size()
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_TOKEN']), triton.cdiv(B, meta['BLOCK_BATCH']))

        torch.cuda.set_device(dx.device)
        rope2d_bwd_kernel[grid](
            dx, idx, row_cs_buffer, col_cs_buffer, 
            dx.stride(0), dx.stride(1), dx.stride(2), 
            row_cs_buffer.stride(0), N_TOTAL, N, N_RDIM)
        
        return dx, None, None, None


class FusedInplaceRotaryPositionEncoding2D(torch.nn.Module):
    @staticmethod
    def encode_idx(height: int, width: int) -> torch.Tensor:
        r"""
            Make token indices for image-like 2D data.

            Args:
                height (Number): the width of given image
                width (Number): the width of given image
        """
        rows = torch.arange(0, height)
        cols = torch.arange(0, width)
        return torch.bitwise_or(rows[:,None] << 16, cols).flatten().to(torch.int32)


    def __init__(self, n_components: int, max_h: int, max_w: int):
        super().__init__()
        self.max_h = max_h
        self.max_w = max_w
        self.row_components = n_components // 2
        self.col_components = n_components // 2

        self.row_buffer = FusedRotaryPositionEncodingContext(self.row_components, max_h)
        self.col_buffer = FusedRotaryPositionEncodingContext(self.col_components, max_w)

    
    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # x: [B, Head, N, C] idx: [N, ]
        B, H, N, C = x.shape
        x0 = x.reshape((-1, N, C))
        return RotaryPE2D.apply(x0, idx, self.row_buffer.cs_components, self.col_buffer.cs_components).reshape(x.shape)


def rope2d_ref(x: torch.Tensor, idx: torch.Tensor, cs_row_buffer: torch.Tensor, cs_col_buffer: torch.Tensor):
    B, N, N_DIM = x.shape
    N_MAX_ROW, N_RDIM, _ = cs_row_buffer.shape

    rx0, rx1 = x[:, :, slice(0, N_RDIM)], x[:, :, slice(N_RDIM,2 * N_RDIM)]
    cx0, cx1 = x[:, :, slice(2 * N_RDIM, 3 * N_RDIM)], x[:, :, slice(3 * N_RDIM,4 * N_RDIM)]
    x_pass = x[:, :, (4 * N_RDIM):]

    idx_row = ((idx >> 16) & 0xFFFF).to(torch.int16) # [N, ]
    idx_col = (idx & 0xFFFF).to(torch.int16)

    one, zero = torch.tensor([[1]]), torch.tensor([[0]])
    row_cos = torch.where(idx_row[:,None] >= 0, cs_row_buffer[torch.maximum(idx_row, torch.tensor([0])), :, 0], one)[None]
    row_sin = torch.where(idx_row[:,None] >= 0, cs_row_buffer[torch.maximum(idx_row, torch.tensor([0])), :, 1], zero)[None]
    col_cos = torch.where(idx_col[:,None] >= 0, cs_col_buffer[torch.maximum(idx_col, torch.tensor([0])), :, 0], one)[None]
    col_sin = torch.where(idx_col[:,None] >= 0, cs_col_buffer[torch.maximum(idx_col, torch.tensor([0])), :, 1], zero)[None]

    ry0, ry1 = rx0 * row_cos - rx1 * row_sin, rx0 * row_sin + rx1 * row_cos
    cy0, cy1 = cx0 * col_cos - cx1 * col_sin, cx0 * col_sin + cx1 * col_cos

    x_result = torch.cat([ry0, ry1, cy0, cy1, x_pass], dim = -1)
    return x_result

if __name__ == "__main__":
    torch.set_default_device('cuda:0')

    B, N, D = 8, 80, 64
    N_RDIM = 8
    H, W = 8, 8

    H_MAX, W_MAX = 8, 8

    row_bases = torch.pow(1000, -2 * (torch.arange(N_RDIM) / N_RDIM))
    row_indices = torch.arange(0, H_MAX)
    row_b_cos_components = torch.cos(torch.outer(row_indices, row_bases))
    row_b_sin_components = torch.sin(torch.outer(row_indices, row_bases))
    row_cs_buffer = torch.dstack([row_b_cos_components, row_b_sin_components])

    col_bases = torch.pow(1000, -2 * (torch.arange(N_RDIM) / N_RDIM))
    col_indices = torch.arange(0, W_MAX)
    col_b_cos_components = torch.cos(torch.outer(col_indices, col_bases))
    col_b_sin_components = torch.sin(torch.outer(col_indices, col_bases))
    col_cs_buffer = torch.dstack([col_b_cos_components, col_b_sin_components])

    C = 128
    x = torch.randn((B, N, C), dtype = torch.float32)

    idx = torch.bitwise_or(torch.arange(0, 8)[:, None] << 16, torch.arange(0, 8)[None, :]).to(torch.int32).flatten()
    idx = torch.concatenate([idx, torch.full((16,),-1, dtype = torch.int32)], dim=-1)
    

    weight = torch.randn((C, D * 2), dtype = torch.float32)

    qk_weight0 = weight.clone().requires_grad_(True)
    qk_weight1 = weight.clone().requires_grad_(True)
    qk0 = (x @ qk_weight0).view((B, N, D, 2))
    qk1 = (x @ qk_weight1).view((B, N, D, 2))
    q0, k0 = qk0[:, :, :, 0], qk0[:, :, :, 1]
    q1, k1 = qk1[:, :, :, 0], qk1[:, :, :, 1]

    q0_o = rope2d_ref(q0, idx, row_cs_buffer, col_cs_buffer)
    q1_o = RotaryPE2D.apply(q1, idx, row_cs_buffer, col_cs_buffer)

    q0_o.flatten().sum().backward()
    q1_o.flatten().sum().backward()

    val_correct = torch.allclose(q0_o, q1_o, 1e-2, 1e-2)
    grad_correct = torch.allclose(qk_weight0.grad, qk_weight1.grad, 1e-2, 1e-2)
    
    if val_correct and grad_correct:
        print('[√] Test passed')