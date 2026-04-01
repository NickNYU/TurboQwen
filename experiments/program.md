# TurboQuant Autoresearch

## Goal
Achieve quality-neutral KV cache compression at minimum bits/channel.

## Metric
Max absolute attention score error across 1000 random Q/K pairs (HEAD_DIM=256).

## Protocol
1. Modify quantizer parameters in turboquant_test.py
2. Run: `python turboquant_test.py`
3. Check metric table
4. Keep if metric improves or holds; discard if worsens

## Baseline
- Float32: 0 error (ground truth)
- QJL 1-bit: ~X error (run to measure)
- TurboQuant 3-bit: ~Y error (target: < 0.01)

## Experiments to try
- Different codebook optimization (Lloyd-Max vs uniform)
- Hadamard vs random rotation
- Residual scaling strategies
- Different bit allocations between MSE and QJL stages
