# TurboQwen: Running a 397B Parameter Model on a Laptop

> **[Read the paper](paper/flash_moe.pdf)** — Full technical details, 90+ experiments, and the story of how an AI and a human built this in 24 hours.

Pure C/Metal inference engine that runs **Qwen3.5-397B-A17B** (a 397 billion parameter Mixture-of-Experts model) on a MacBook Pro with 48GB RAM at **4.4+ tokens/second** with production-quality output including tool calling.

The entire 209GB model streams from SSD through a custom Metal compute pipeline. No Python. No frameworks. Just C, Objective-C, and hand-tuned Metal shaders.

## Results

![Progress](progress.png)

| Configuration | tok/s | Quality | Notes |
|--------------|-------|---------|-------|
| 4-bit experts, FMA kernel | **4.36** | Excellent | Current best. Full tool calling. 209GB on disk. |
| 4-bit Hadamard g256 | **TBD** | Excellent | 8.3% smaller experts, reduced I/O overhead. |
| 4-bit experts, baseline | 3.90 | Excellent | Before FMA kernel optimization. |
| 2-bit experts, trust OS | 5.74 | Good* | 120GB on disk. *Breaks JSON/tool calling. |
| 2-bit peak single token | 7.05 | Good* | Warm cache burst. *Not suitable for tool use. |

**KV cache compression** (M1 Max 32GB, Qwen3.5-35B-A3B, 100 tokens):

| KV Config | tok/s | Compression | Quality | Notes |
|-----------|-------|-------------|---------|-------|
| float32 (default) | **12.27** | 1x | Baseline | Full precision K+V cache |
| TurboQuant 2-bit (`--tq 2`) | **11.57** | 15.5x | Near-perfect | 1-bit MSE + 1-bit QJL residual |
| TurboQuant 3-bit (`--tq 3`) | **10.34** | 10.4x | Quality-neutral | 2-bit MSE + 1-bit QJL residual. **Recommended.** |
| TurboQuant 4-bit (`--tq 4`) | **11.32** | 7.9x | Quality-neutral | 3-bit MSE + 1-bit QJL residual |
| QJL 1-bit (`--qjl`) | **11.28** | 32x | Good | Legacy. TQ-2 is faster and better quality. |

*TurboQuant implements [Zandieh et al., 2025]: Hadamard rotation + Lloyd-Max MSE quantizer + QJL residual correction. At 3 bits/channel, attention KL divergence is effectively zero.*

## Hardware

- **Machine**: MacBook Pro, Apple M3 Max
- **Chip**: 16-core CPU (12P + 4E), 40-core GPU, 16-core ANE
- **Memory**: 48 GB unified (~400 GB/s bandwidth)
- **SSD**: 1TB Apple Fabric, **17.5 GB/s sequential read** (measured)
- **macOS**: 26.2 (Darwin 25.2.0)

## Architecture

The model has 60 transformer layers: 45 GatedDeltaNet (linear attention) + 15 standard full attention. Each layer has 512 experts, of which K=4 are activated per token (plus one shared expert). Hidden dimension is 4096.

### Key Techniques

1. **SSD Expert Streaming** — Expert weights (209GB at 4-bit) are read from NVMe SSD on demand via parallel `pread()` with GCD dispatch groups. Only the K=4 active experts per layer are loaded (~6.75MB each). The OS page cache manages caching — no custom cache needed ("Trust the OS" principle). Inspired by Apple's "LLM in a Flash" paper.

2. **FMA-Optimized Dequant Kernel** — The inner loop of the 4-bit dequantized matrix-vector multiply rearranges the math from `(nibble * scale + bias) * x` to `fma(nibble, scale*x, bias*x)`. Pre-computing `scale*x` and `bias*x` lets the GPU fused multiply-add unit do dequant+multiply in one instruction. 12% faster than the naive formulation.

3. **Hadamard Rotation + group_size=256** — Expert weights are rotated by a Hadamard matrix to smooth outliers, enabling 4x larger quantization groups (256 vs 64). Reduces scale/bias overhead from 11.1% to ~3%, saving 12.8GB across all experts. The inverse Hadamard is applied to input vectors at runtime via an O(n log n) butterfly transform in the Metal kernel — zero overhead when disabled.

4. **Metal Compute Shaders** — Hand-written Metal kernels for:
   - 4-bit and 2-bit dequantized matrix-vector multiply (tiled, SIMD-reduced, shared input cache, FMA-optimized)
   - In-kernel Walsh-Hadamard butterfly transform (gated, zero-cost when off)
   - Fused SwiGLU activation
   - RMS normalization (two-pass: sum-of-squares reduction + apply)
   - Batched GPU attention (Q@K^T, softmax, scores@V) for full attention layers
   - GPU RoPE (fused with Q deinterleave and K normalization)
   - MoE combine + residual + sigmoid gate (fused kernel)

5. **Deferred GPU Expert Compute** — CMD3 (expert forward pass) is submitted without waiting. The GPU executes it while the CPU prepares the next layer. The combine + residual + norm are also on GPU, feeding directly into the next layer's attention projections.

6. **Accelerate BLAS for Linear Attention** — The GatedDeltaNet recurrence uses `cblas_sscal`, `cblas_sgemv`, and `cblas_sger` for the 64-head × 128×128 state matrix update. 64% faster than scalar code.

7. **TurboQuant KV Cache Compression** — Full-attention layers use TurboQuant ([Zandieh et al., 2025](https://arxiv.org/abs/2504.19874)) to compress K vectors: Hadamard rotation smooths coordinates, then a Lloyd-Max MSE quantizer captures magnitude (b-1 bits), and QJL sign projection encodes the residual (1 bit). At 3 bits/channel (`--tq 3`), achieves 10.4x K compression with effectively zero attention KL divergence. Supersedes the legacy `--qjl` 1-bit mode.

8. **Trust the OS** — No custom expert cache. The OS page cache (~35GB) manages expert data caching via standard LRU. Every custom caching approach we tested (Metal LRU, malloc cache, LZ4 compressed cache) was slower due to GPU memory pressure or overhead. The page cache achieves ~71% hit rate naturally.

### Pipeline Per Layer (4.28ms average at 4-bit)

```
CMD3(prev) → CMD1: attention projections + delta-net  [1.22ms GPU]
           → CPU: flush results                       [0.01ms CPU]
           → CMD2: o_proj + norm + routing + shared    [0.55ms GPU]
           → CPU: softmax + topK routing               [0.003ms]
           → I/O: parallel pread K=4 experts           [2.41ms SSD]
           → CMD3: expert forward + combine + norm     [0.04ms encode, DEFERRED]
```

### Unified Memory Constraint

On Apple Silicon, SSD DMA and GPU compute share the same memory controller and cannot be profitably overlapped. The GPU's dequant kernels are bandwidth-saturated at ~418 GiB/s. Even small background SSD DMA causes disproportionate GPU latency spikes through memory controller arbitration. The serial pipeline (GPU → SSD → GPU) is hardware-optimal.

## Quick Start

```bash
cd metal_infer
make

# 4-bit inference
./infer --prompt "Explain quantum computing" --tokens 100

# Hadamard g256 experts (auto-detected if packed_experts_g256/ exists)
./infer --g256 --prompt "Explain quantum computing" --tokens 100

# 2-bit inference (faster but breaks tool calling)
./infer --prompt "Explain quantum computing" --tokens 100 --2bit

# TurboQuant KV cache compression (recommended for long context)
./infer --tq 3 --prompt "Explain quantum computing" --tokens 100

# QJL 1-bit KV cache compression (legacy, use --tq instead)
./infer --qjl --prompt "Explain quantum computing" --tokens 100

# Interactive chat with tool calling
./chat

# Per-layer timing breakdown
./infer --prompt "Hello" --tokens 20 --timing
```

### Preparing Hadamard g256 Experts

```bash
# Repack all layers (reads packed_experts/, writes packed_experts_g256/)
python repack_experts_hadamard.py

# Repack one layer with verification
python repack_experts_hadamard.py --layers 0 --verify

# Repack specific layers
python repack_experts_hadamard.py --layers 0-4
```

## Project Structure

```
metal_infer/
  infer.m              # Complete inference engine (~7000 lines)
  shaders.metal        # Metal compute kernels (~1200 lines)
  chat.m               # Interactive chat TUI with tool calling
  tokenizer.h          # C BPE tokenizer (single-header, 449 lines)
  main.m               # MoE-only benchmark
  Makefile             # Build system
  extract_weights.py   # Creates model_weights.bin from safetensors
  repack_experts_2bit.py  # 4-bit → 2-bit expert requantization

repack_experts.py              # 4-bit expert packing from safetensors
repack_experts_hadamard.py     # Hadamard rotation + group_size=256 repacking
experiments/
  codebook.py                  # Lloyd-Max codebook precomputation for TurboQuant
  turboquant_test.py           # TurboQuant quality benchmark (float32/QJL/TQ comparison)
  program.md                   # Autoresearch agent instructions
progress.py                    # Results visualization (Q2/Q4 tracks)
results.tsv                    # Experiment log (58 experiments)
paper/                         # LaTeX source + PDF of the paper
```

## What We Tried (and What Worked)

### Kept
| Approach | Result | Impact |
|----------|--------|--------|
| FMA dequant kernel | GPU compute -12% | **+12% tok/s** |
| Hadamard g256 | Expert size -8.3% | **-12.8GB total, faster I/O** |
| TurboQuant KV cache | 10.4x K, quality-neutral | **Long context, replaces QJL** |
| QJL 1-bit KV cache | 32x K compression | **Long context (legacy)** |
| Trust OS page cache | Deleted Metal LRU → +38% | **Foundational** |
| GPU combine+norm in CMD3 | Eliminates CPU round-trip | **Pipeline** |
| BLAS delta-net (Accelerate) | cpu_attn 0.78→0.28ms | **+64% attn** |
| F_NOCACHE for 2-bit | +3% from avoiding page thrash | **2-bit only** |
| GPU fused attention (RoPE) | +2% for full-attn layers | **Small** |
| C BPE tokenizer | 180ms vs 3500ms startup | **20x startup** |
| Deferred CMD3 execution | GPU/CPU overlap | **Pipeline** |

### Discarded (58 experiments, highlights)
| Approach | Result | Why |
|----------|--------|-----|
| LZ4 expert compression | -13% | Decompress overhead > warm cache savings |
| F_RDADVISE prefetch | net 0% | Unified memory: SSD DMA slows GPU -73% |
| Temporal expert prediction | -18% | 25% hit rate, SSD bandwidth waste |
| MLP routing predictor | 31% accuracy | Worse than temporal baseline |
| GPU LUT dequant kernel | -2% | Indirect register access serializes |
| GPU private buffer compression | -20% pipeline | Blit cost 4×7MB > matvec savings |
| Spin-poll GPU wait | -23% | CPU thermal competes with GPU |
| Expert file clustering | 0% | NVMe ignores scatter at 7MB granularity |
| dispatch_io | -70% | dispatch_data management overhead |
| mmap expert files | -5x | Per-page fault overhead on cold data |
| Speculative early routing | -38% | Cache pollution + overhead |
| MTP speculative decoding | break-even | MoE I/O scales per-token (unlike dense) |

## Safety

This is a primary development machine. The engine explicitly controls memory:
- Non-expert weights: 5.5GB (mmap'd, read-only)
- Metal scratch buffers: ~200MB
- Total: ~6GB, leaving 42GB for OS + page cache
- No OOM risk. Expert data streams from SSD on demand.
- No custom caches. Trust the OS.

## License

MIT
