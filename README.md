# A deep learning template library

~~ As is well known, deep learning requires cards and cash capabilities (lol). But what if I only have some 2080Ti(s)... ~~

Here is a deep learning template library collecting models/modules/ops written for practice. 

Content:

- A SM75 (may be lower) compatible FlashAttention triton implementation. Support different qk/vo dimension to support cross attention. Based on triton fused attention tutorial (fixed non-causal gradient error)
- A fused 2D rotary positional encoding function. Allow specifying the row or column corresponding to each token, or no positional encoding for certain token. Allow applying positional encoding only in the first k dimensions of the query.
- A tiled 2D slide window FlashAttention implementation. x2~4 better performance than natten (evaluated at equivalent receptive field size) in both fwd/bwd passes. Compatible with fused RoPE-2D above.

TODO:

- A grouped SwiGLU ops with multiple in-group experts
- A chunkwise Gated DeltaNet ops
