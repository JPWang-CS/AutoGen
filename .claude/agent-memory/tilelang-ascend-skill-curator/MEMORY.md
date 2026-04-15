# TileLang-Ascend Skill Curator Memory

## Key API Patterns Verified (2026-04-14)

### GEMM API: T.gemm_v0 (NOT T.gemm)
- `T.gemm_v0(A_L1, B_L1, C_L0, init=(k==0))` -- init=True clears C before compute
- `T.mma(A_L0A, B_L0B, C_L0, init=...)` -- lower level, needs L0A/L0B

### Memory APIs: Expert vs Developer
- Expert: `T.alloc_L1`, `T.alloc_L0C`, `T.alloc_ub`, `T.Scope("C"/"V")`
- Developer: `T.alloc_shared`, `T.alloc_fragment`, no Scope needed

### Pipeline APIs
- `T.Pipelined(range, num_stages=N)` -- auto pipeline, no nesting!
- `T.set_flag`/`T.wait_flag` -- manual intra-core pipeline
- `T.set_cross_flag`/`T.wait_cross_flag` -- inter-core Cube<->Vector sync

### Persistent API
- `T.Persistent([m_num, n_num], core_num, cid)` -- iterate over blocks with fixed cores

### Swizzle
- `T.use_swizzle(idx, M, N, K, block_M, block_N, off=3)` -- block scheduling

### AutoTune
- `@tilelang.autotune(configs=..., ref_prog=..., supply_prog=...)` before `@tilelang.jit`
- `carver.MatmulTemplate(...).with_arch(Ascend()).recommend_hints(topk=N)` -- hardware-aware config

## Skill Library Status (2026-04-14)

### tl-op-cube additions
- Added: Batch GEMM Expert mode template
- Added: Conv2D complete template with im2col
- Added: Grouped GEMM (variable batch) template
- Added: Pipelined GEMM (T.Pipelined) template
- Added: Tail Block Developer mode (auto with T.ceildiv)
- Added: AutoTune and Carver templates
- Added: Performance optimization section (block sizes, pipeline strategies, perf data)

### tl-op-fused additions
- Added: Matmul+Add Pipelined (T.Pipelined for inter-core)
- Added: Quantized Batch Matmul (int8->int32->scale->fp16)
- Added: FA performance data (80% of native AscendC)
- Added: num_stages tuning guide
- Added: Sync frequency optimization guide

### tl-op-vector additions
- Added: Performance optimization section
- Added: VEC_NUM selection guide
- Added: Block size tuning guide with UB capacity calculation
- Added: Pipeline optimization (MTE2-V-MTE3)
- Added: Scalar vectorization pattern
- Added: Instruction merging (axpy) pattern

## Performance Notes
- FA Expert: 80% native | Developer: 60% native
- Key optimizations: L1 residency, vectorization, multi-buffer, sync elimination, pipelined, reduced sync, axpy
- Default block config: 128/256/64
- core_num for intrinsic: 20 (910B)
- Persistent core_num: 24
