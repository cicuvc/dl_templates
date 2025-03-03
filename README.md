# A deep learning template library

~~ As is well known, deep learning requires cards and cash capabilities (lol). But what if I only have some 2080Ti(s)... ~~

Here is a deep learning template library collecting models/modules/ops written for practice. 

Content:

- A SM75 (may be lower) compatible FlashAttention triton implementation. Support different qk/vo dimension to support cross attention. Based on triton fused attention tutorial (fixed non-causal gradient error)

TODO:

- A chunkwise Gated DeltaNet ops
- Some fused rotary position embedding 