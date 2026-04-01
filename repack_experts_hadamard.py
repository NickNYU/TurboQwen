#!/usr/bin/env python3
"""Repack 4-bit experts with Hadamard rotation and group_size=256.

Reads from packed_experts/layer_XX.bin (group_size=64, no rotation).
Writes to packed_experts_g256/layer_XX.bin (group_size=256, Hadamard-rotated).

Per expert:
  1. Read packed 4-bit weights + bf16 scales/biases (group_size=64 layout)
  2. Dequantize to float32
  3. Apply Hadamard rotation: W_rot = W @ H (normalized)
  4. Re-quantize to 4-bit affine with group_size=256
  5. Pack into new contiguous layout

New layout per expert (1,622,144 bytes):
  gate_proj.weight  [512, 256]  u32    offset 0
  gate_proj.scales  [512, 8]    bf16   offset 524288
  gate_proj.biases  [512, 8]    bf16   offset 532480
  up_proj.weight    [512, 256]  u32    offset 540672
  up_proj.scales    [512, 8]    bf16   offset 1064960
  up_proj.biases    [512, 8]    bf16   offset 1073152
  down_proj.weight  [2048, 64]  u32    offset 1081344
  down_proj.scales  [2048, 2]   bf16   offset 1605632
  down_proj.biases  [2048, 2]   bf16   offset 1613824

Usage:
    python repack_experts_hadamard.py                    # repack all 40 layers
    python repack_experts_hadamard.py --layers 0         # repack layer 0 only
    python repack_experts_hadamard.py --layers 0 --verify  # repack + verify roundtrip
    python repack_experts_hadamard.py --model-path /path/to/model
"""

import argparse
import os
import struct
import sys
import time
import numpy as np

# ============================================================================
# Constants — original layout (group_size=64)
# ============================================================================

EXPERT_SIZE_OLD = 1769472
NUM_EXPERTS = 256
NUM_LAYERS = 40

# Original component layout (group_size=64)
COMPONENTS_OLD = [
    {"name": "gate_proj.weight", "offset": 0,       "size": 524288,  "shape": (512, 256),  "out_dim": 512,  "in_dim": 2048},
    {"name": "gate_proj.scales", "offset": 524288,   "size": 32768,   "shape": (512, 32)},
    {"name": "gate_proj.biases", "offset": 557056,   "size": 32768,   "shape": (512, 32)},
    {"name": "up_proj.weight",   "offset": 589824,   "size": 524288,  "shape": (512, 256),  "out_dim": 512,  "in_dim": 2048},
    {"name": "up_proj.scales",   "offset": 1114112,  "size": 32768,   "shape": (512, 32)},
    {"name": "up_proj.biases",   "offset": 1146880,  "size": 32768,   "shape": (512, 32)},
    {"name": "down_proj.weight", "offset": 1179648,  "size": 524288,  "shape": (2048, 64),  "out_dim": 2048, "in_dim": 512},
    {"name": "down_proj.scales", "offset": 1703936,  "size": 32768,   "shape": (2048, 8)},
    {"name": "down_proj.biases", "offset": 1736704,  "size": 32768,   "shape": (2048, 8)},
]

# ============================================================================
# Constants — new layout (group_size=256)
# ============================================================================

EXPERT_SIZE_NEW = 1622144

COMPONENTS_NEW = [
    {"name": "gate_proj.weight", "offset": 0,        "size": 524288,  "shape": (512, 256)},
    {"name": "gate_proj.scales", "offset": 524288,    "size": 8192,    "shape": (512, 8)},
    {"name": "gate_proj.biases", "offset": 532480,    "size": 8192,    "shape": (512, 8)},
    {"name": "up_proj.weight",   "offset": 540672,    "size": 524288,  "shape": (512, 256)},
    {"name": "up_proj.scales",   "offset": 1064960,   "size": 8192,    "shape": (512, 8)},
    {"name": "up_proj.biases",   "offset": 1073152,   "size": 8192,    "shape": (512, 8)},
    {"name": "down_proj.weight", "offset": 1081344,   "size": 524288,  "shape": (2048, 64)},
    {"name": "down_proj.scales", "offset": 1605632,   "size": 8192,    "shape": (2048, 2)},  # 512/256 = 2 groups
    {"name": "down_proj.biases", "offset": 1613824,   "size": 8192,    "shape": (2048, 2)},
]

GROUP_SIZE_OLD = 64
GROUP_SIZE_NEW = 256


# ============================================================================
# bf16 helpers
# ============================================================================

def bf16_to_f32(bf16_arr):
    """Convert array of uint16 (bf16) to float32."""
    # bf16 is just the top 16 bits of float32
    u32 = bf16_arr.astype(np.uint32) << 16
    return u32.view(np.float32)


def f32_to_bf16(f32_arr):
    """Convert float32 array to uint16 (bf16) via truncation."""
    u32 = f32_arr.view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


# ============================================================================
# Dequantize 4-bit
# ============================================================================

def dequant_4bit(packed_u32, scales_bf16, biases_bf16, out_dim, in_dim, group_size):
    """Dequantize 4-bit packed weights to float32.

    packed_u32: [out_dim, in_dim/8] uint32
    scales_bf16: [out_dim, in_dim/group_size] uint16 (bf16)
    biases_bf16: [out_dim, in_dim/group_size] uint16 (bf16)
    Returns: [out_dim, in_dim] float32
    """
    scales = bf16_to_f32(scales_bf16)  # [out_dim, num_groups]
    biases = bf16_to_f32(biases_bf16)

    # Unpack nibbles: each uint32 holds 8 x 4-bit values
    # packed_u32 shape: [out_dim, in_dim/8]
    nibbles = np.zeros((out_dim, in_dim), dtype=np.float32)
    for n in range(8):
        nibbles[:, n::8] = ((packed_u32 >> (n * 4)) & 0xF).astype(np.float32)

    # Apply scale and bias per group
    num_groups = in_dim // group_size
    result = np.zeros((out_dim, in_dim), dtype=np.float32)
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        result[:, start:end] = nibbles[:, start:end] * scales[:, g:g+1] + biases[:, g:g+1]

    return result


# ============================================================================
# Quantize 4-bit
# ============================================================================

def quant_4bit(weights_f32, group_size):
    """Quantize float32 weights to 4-bit affine with given group_size.

    weights_f32: [out_dim, in_dim] float32
    Returns: (packed_u32, scales_bf16, biases_bf16)
    """
    out_dim, in_dim = weights_f32.shape
    num_groups = in_dim // group_size
    packed_cols = in_dim // 8

    scales = np.zeros((out_dim, num_groups), dtype=np.float32)
    biases = np.zeros((out_dim, num_groups), dtype=np.float32)
    nibbles = np.zeros((out_dim, in_dim), dtype=np.uint8)

    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        group = weights_f32[:, start:end]

        gmin = group.min(axis=1, keepdims=True)
        gmax = group.max(axis=1, keepdims=True)

        # scale = (max - min) / 15, bias = min
        s = (gmax - gmin) / 15.0
        # Avoid division by zero for constant groups
        s = np.where(s < 1e-10, 1e-10, s)
        b = gmin

        scales[:, g] = s.squeeze()
        biases[:, g] = b.squeeze()

        # Quantize: nibble = round((value - bias) / scale), clamp to [0, 15]
        quantized = np.round((group - b) / s).clip(0, 15).astype(np.uint8)
        nibbles[:, start:end] = quantized

    # Pack nibbles into uint32 (8 nibbles per uint32)
    packed = np.zeros((out_dim, packed_cols), dtype=np.uint32)
    for n in range(8):
        packed |= nibbles[:, n::8].astype(np.uint32) << (n * 4)

    return packed, f32_to_bf16(scales), f32_to_bf16(biases)


# ============================================================================
# Hadamard transform
# ============================================================================

def hadamard_matrix(dim):
    """Generate normalized Hadamard matrix of size dim x dim.

    Uses the Sylvester construction (recursive Kronecker product).
    dim must be a power of 2.
    """
    assert dim > 0 and (dim & (dim - 1)) == 0, f"dim must be power of 2, got {dim}"
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < dim:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(dim)


# Pre-compute Hadamard matrices for the two dimensions we need
_H_2048 = None
_H_512 = None

def get_hadamard(dim):
    global _H_2048, _H_512
    if dim == 2048:
        if _H_2048 is None:
            print("  Computing H_2048...", end="", flush=True)
            _H_2048 = hadamard_matrix(2048)
            print(" done")
        return _H_2048
    elif dim == 512:
        if _H_512 is None:
            print("  Computing H_512...", end="", flush=True)
            _H_512 = hadamard_matrix(512)
            print(" done")
        return _H_512
    else:
        return hadamard_matrix(dim)


# ============================================================================
# Core: repack one expert
# ============================================================================

def repack_expert(expert_data):
    """Repack one expert: dequant -> Hadamard rotate -> requant -> pack.

    expert_data: bytes of length EXPERT_SIZE_OLD
    Returns: bytes of length EXPERT_SIZE_NEW
    """
    assert len(expert_data) == EXPERT_SIZE_OLD

    output = bytearray(EXPERT_SIZE_NEW)

    # Process each projection: gate, up, down
    projections = [
        ("gate_proj", 512, 2048,
         0, 524288, 557056,            # old weight/scale/bias offsets
         0, 524288, 532480),           # new weight/scale/bias offsets
        ("up_proj", 512, 2048,
         589824, 1114112, 1146880,
         540672, 1064960, 1073152),
        ("down_proj", 2048, 512,
         1179648, 1703936, 1736704,
         1081344, 1605632, 1613824),
    ]

    for name, out_dim, in_dim, old_w, old_s, old_b, new_w, new_s, new_b in projections:
        packed_cols = in_dim // 8
        old_num_groups = in_dim // GROUP_SIZE_OLD

        # Read old packed weights
        w_bytes = expert_data[old_w:old_w + out_dim * packed_cols * 4]
        packed_u32 = np.frombuffer(w_bytes, dtype=np.uint32).reshape(out_dim, packed_cols)

        # Read old scales/biases (bf16)
        s_bytes = expert_data[old_s:old_s + out_dim * old_num_groups * 2]
        scales_bf16 = np.frombuffer(s_bytes, dtype=np.uint16).reshape(out_dim, old_num_groups)
        b_bytes = expert_data[old_b:old_b + out_dim * old_num_groups * 2]
        biases_bf16 = np.frombuffer(b_bytes, dtype=np.uint16).reshape(out_dim, old_num_groups)

        # Dequantize to float32
        W_f32 = dequant_4bit(packed_u32, scales_bf16, biases_bf16, out_dim, in_dim, GROUP_SIZE_OLD)

        # Apply Hadamard rotation: W_rot = W @ H
        H = get_hadamard(in_dim)
        W_rot = W_f32 @ H

        # Re-quantize with group_size=256
        new_packed, new_scales, new_biases = quant_4bit(W_rot, GROUP_SIZE_NEW)

        # Write into output buffer
        output[new_w:new_w + new_packed.nbytes] = new_packed.tobytes()
        output[new_s:new_s + new_scales.nbytes] = new_scales.tobytes()
        output[new_b:new_b + new_biases.nbytes] = new_biases.tobytes()

    return bytes(output)


# ============================================================================
# Verify roundtrip quality
# ============================================================================

def verify_expert(old_data, new_data):
    """Check that old -> dequant -> rotate -> requant -> new -> dequant -> unrotate ≈ old dequant."""
    projections = [
        ("gate_proj", 512, 2048, 0, 524288, 557056, 0, 524288, 532480),
        ("up_proj", 512, 2048, 589824, 1114112, 1146880, 540672, 1064960, 1073152),
        ("down_proj", 2048, 512, 1179648, 1703936, 1736704, 1081344, 1605632, 1613824),
    ]

    for name, out_dim, in_dim, old_w, old_s, old_b, new_w, new_s, new_b in projections:
        packed_cols = in_dim // 8
        old_num_groups = in_dim // GROUP_SIZE_OLD
        new_num_groups = in_dim // GROUP_SIZE_NEW

        # Dequant original
        w_old = np.frombuffer(old_data[old_w:old_w + out_dim * packed_cols * 4], dtype=np.uint32).reshape(out_dim, packed_cols)
        s_old = np.frombuffer(old_data[old_s:old_s + out_dim * old_num_groups * 2], dtype=np.uint16).reshape(out_dim, old_num_groups)
        b_old = np.frombuffer(old_data[old_b:old_b + out_dim * old_num_groups * 2], dtype=np.uint16).reshape(out_dim, old_num_groups)
        W_orig = dequant_4bit(w_old, s_old, b_old, out_dim, in_dim, GROUP_SIZE_OLD)

        # Dequant new + inverse Hadamard
        w_new = np.frombuffer(new_data[new_w:new_w + out_dim * packed_cols * 4], dtype=np.uint32).reshape(out_dim, packed_cols)
        s_new = np.frombuffer(new_data[new_s:new_s + out_dim * new_num_groups * 2], dtype=np.uint16).reshape(out_dim, new_num_groups)
        b_new = np.frombuffer(new_data[new_b:new_b + out_dim * new_num_groups * 2], dtype=np.uint16).reshape(out_dim, new_num_groups)
        W_rot = dequant_4bit(w_new, s_new, b_new, out_dim, in_dim, GROUP_SIZE_NEW)

        # Inverse Hadamard: H is its own inverse (H @ H = I for normalized H)
        H = get_hadamard(in_dim)
        W_recovered = W_rot @ H

        # Compare
        err = np.abs(W_recovered - W_orig)
        rmse = np.sqrt(np.mean(err**2))
        max_err = err.max()
        orig_rms = np.sqrt(np.mean(W_orig**2))
        rel_err = rmse / (orig_rms + 1e-10)

        print(f"    {name}: RMSE={rmse:.6f}, max_err={max_err:.6f}, "
              f"rel_err={rel_err:.4%}, orig_rms={orig_rms:.6f}")


# ============================================================================
# CLI
# ============================================================================

def parse_layers(spec):
    if spec is None or spec == 'all':
        return list(range(NUM_LAYERS))
    layers = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def main():
    parser = argparse.ArgumentParser(
        description="Repack 4-bit experts with Hadamard rotation and group_size=256")
    parser.add_argument('--model-path', default='./model',
                        help='Path to model directory (default: ./model)')
    parser.add_argument('--layers', default=None,
                        help='Layer spec: "all", "0-4", "0,5,10" (default: all)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify roundtrip quality after repacking')
    parser.add_argument('--dry-run', action='store_true',
                        help='Check source files exist without writing')
    args = parser.parse_args()

    input_dir = os.path.join(args.model_path, "packed_experts")
    output_dir = os.path.join(args.model_path, "packed_experts_g256")

    layers = parse_layers(args.layers)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Layers: {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"Old expert size: {EXPERT_SIZE_OLD:,} bytes (group_size={GROUP_SIZE_OLD})")
    print(f"New expert size: {EXPERT_SIZE_NEW:,} bytes (group_size={GROUP_SIZE_NEW})")
    print(f"Savings: {EXPERT_SIZE_OLD - EXPERT_SIZE_NEW:,} bytes/expert "
          f"({(1 - EXPERT_SIZE_NEW/EXPERT_SIZE_OLD)*100:.1f}%)")

    # Check input files exist
    for layer_idx in layers:
        path = os.path.join(input_dir, f"layer_{layer_idx:02d}.bin")
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            sys.exit(1)

    if args.dry_run:
        print("DRY RUN: all source files found")
        return

    os.makedirs(output_dir, exist_ok=True)

    old_layer_size = NUM_EXPERTS * EXPERT_SIZE_OLD
    new_layer_size = NUM_EXPERTS * EXPERT_SIZE_NEW

    t_start = time.monotonic()
    total_written = 0

    for i, layer_idx in enumerate(layers):
        t_layer = time.monotonic()
        in_path = os.path.join(input_dir, f"layer_{layer_idx:02d}.bin")
        out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

        print(f"\nLayer {layer_idx:2d}:")

        # Read entire layer file
        with open(in_path, 'rb') as f:
            layer_data = f.read()
        assert len(layer_data) == old_layer_size, \
            f"Expected {old_layer_size}, got {len(layer_data)}"

        # Pre-allocate output
        fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd_out, new_layer_size)

        for expert_idx in range(NUM_EXPERTS):
            old_off = expert_idx * EXPERT_SIZE_OLD
            expert_data = layer_data[old_off:old_off + EXPERT_SIZE_OLD]

            new_expert = repack_expert(expert_data)
            assert len(new_expert) == EXPERT_SIZE_NEW

            new_off = expert_idx * EXPERT_SIZE_NEW
            os.pwrite(fd_out, new_expert, new_off)

            if (expert_idx + 1) % 32 == 0 or expert_idx == NUM_EXPERTS - 1:
                elapsed = time.monotonic() - t_layer
                eta_layer = elapsed / (expert_idx + 1) * (NUM_EXPERTS - expert_idx - 1)
                print(f"  experts {expert_idx+1:3d}/{NUM_EXPERTS} "
                      f"({elapsed:.1f}s, ETA {eta_layer:.0f}s)", end="\r")

        os.close(fd_out)
        elapsed = time.monotonic() - t_layer
        total_written += new_layer_size
        overall_elapsed = time.monotonic() - t_start
        print(f"  Layer {layer_idx:2d}: {new_layer_size/1024**3:.2f} GB in {elapsed:.1f}s | "
              f"Total: {total_written/1024**3:.1f}/{len(layers)*new_layer_size/1024**3:.1f} GB")

        # Verify if requested
        if args.verify:
            print(f"  Verifying layer {layer_idx}...")
            with open(out_path, 'rb') as f:
                new_layer = f.read()
            # Spot check experts 0 and 127
            for eidx in [0, 127]:
                print(f"  Expert {eidx}:")
                old_expert = layer_data[eidx * EXPERT_SIZE_OLD:(eidx + 1) * EXPERT_SIZE_OLD]
                new_expert = new_layer[eidx * EXPERT_SIZE_NEW:(eidx + 1) * EXPERT_SIZE_NEW]
                verify_expert(old_expert, new_expert)

    total_elapsed = time.monotonic() - t_start
    print(f"\n{'='*60}")
    print(f"DONE: {total_written:,} bytes ({total_written/1024**3:.1f} GB) written")
    print(f"Time: {total_elapsed:.1f}s")
    print(f"Output: {output_dir}")
    print(f"Layer file size: {new_layer_size:,} bytes "
          f"(was {old_layer_size:,}, -{(1-new_layer_size/old_layer_size)*100:.1f}%)")


if __name__ == '__main__':
    main()
