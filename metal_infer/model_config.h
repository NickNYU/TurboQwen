/*
 * model_config.h — Build-time model configuration for TurboQwen inference engine
 *
 * Supports Qwen3.5-35B-A3B (default) and Qwen3.5-397B-A17B (MODEL_397B).
 *
 * Usage:
 *   make                  # builds for 35B (default)
 *   make MODEL=397B       # builds for 397B
 *
 * Both models share: HEAD_DIM=256, NUM_ATTN_HEADS=16, NUM_KV_HEADS=2,
 * VOCAB_SIZE=248320, FULL_ATTN_INTERVAL=4, GROUP_SIZE=64, CONV_KERNEL_SIZE=4.
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

// ============================================================================
// Model-specific constants
// ============================================================================

#ifdef MODEL_397B

// Qwen3.5-397B-A17B
#define HIDDEN_DIM          4096
#define NUM_LAYERS          60
#define NUM_EXPERTS         512
#define NUM_EXPERTS_PER_TOK 4
#define MOE_INTERMEDIATE    1024
#define SHARED_INTERMEDIATE 1024

// Linear attention (GatedDeltaNet) head counts
#define LINEAR_NUM_V_HEADS  64
#define LINEAR_NUM_K_HEADS  32

// 4-bit expert packed binary layout (from main.m verified layout)
#define EXPERT_SIZE         7077888
#define GATE_W_OFF  0
#define GATE_S_OFF  2097152
#define GATE_B_OFF  2228224
#define UP_W_OFF    2359296
#define UP_S_OFF    4456448
#define UP_B_OFF    4587520
#define DOWN_W_OFF  4718592
#define DOWN_S_OFF  6815744
#define DOWN_B_OFF  6946816

// 2-bit and g256 layouts not yet supported for 397B
// Building with --2bit or --g256 will use these zero-value placeholders
// and fail at runtime with a size mismatch — this is intentional.
#define EXPERT_SIZE_2BIT    0
#define GATE_W_OFF_2  0
#define GATE_S_OFF_2  0
#define GATE_B_OFF_2  0
#define UP_W_OFF_2    0
#define UP_S_OFF_2    0
#define UP_B_OFF_2    0
#define DOWN_W_OFF_2  0
#define DOWN_S_OFF_2  0
#define DOWN_B_OFF_2  0

#define EXPERT_SIZE_G256    0
#define GATE_W_OFF_G256     0
#define GATE_S_OFF_G256     0
#define GATE_B_OFF_G256     0
#define UP_W_OFF_G256       0
#define UP_S_OFF_G256       0
#define UP_B_OFF_G256       0
#define DOWN_W_OFF_G256     0
#define DOWN_S_OFF_G256     0
#define DOWN_B_OFF_G256     0

#else

// Qwen3.5-35B-A3B (default)
#define HIDDEN_DIM          2048
#define NUM_LAYERS          40
#define NUM_EXPERTS         256
#define NUM_EXPERTS_PER_TOK 8
#define MOE_INTERMEDIATE    512
#define SHARED_INTERMEDIATE 512

// Linear attention (GatedDeltaNet) head counts
#define LINEAR_NUM_V_HEADS  32
#define LINEAR_NUM_K_HEADS  16

// 4-bit expert packed binary layout
#define EXPERT_SIZE         1769472
#define GATE_W_OFF  0
#define GATE_S_OFF  524288
#define GATE_B_OFF  557056
#define UP_W_OFF    589824
#define UP_S_OFF    1114112
#define UP_B_OFF    1146880
#define DOWN_W_OFF  1179648
#define DOWN_S_OFF  1703936
#define DOWN_B_OFF  1736704

// 2-bit expert layout (from repack_experts_2bit.py)
#define EXPERT_SIZE_2BIT    983040
#define GATE_W_OFF_2  0
#define GATE_S_OFF_2  262144
#define GATE_B_OFF_2  294912
#define UP_W_OFF_2    327680
#define UP_S_OFF_2    589824
#define UP_B_OFF_2    622592
#define DOWN_W_OFF_2  655360
#define DOWN_S_OFF_2  917504
#define DOWN_B_OFF_2  950272

// Hadamard g256 expert layout (group_size=256, Hadamard-rotated weights)
#define EXPERT_SIZE_G256    1622144
#define GATE_W_OFF_G256     0
#define GATE_S_OFF_G256     524288
#define GATE_B_OFF_G256     532480
#define UP_W_OFF_G256       540672
#define UP_S_OFF_G256       1064960
#define UP_B_OFF_G256       1073152
#define DOWN_W_OFF_G256     1081344
#define DOWN_S_OFF_G256     1605632
#define DOWN_B_OFF_G256     1613824

#endif // MODEL_397B

// ============================================================================
// Shared constants (identical for both models)
// ============================================================================

#define NUM_ATTN_HEADS      16
#define NUM_KV_HEADS        2
#define HEAD_DIM            256
#define VOCAB_SIZE          248320
#define RMS_NORM_EPS        1e-6f
#define FULL_ATTN_INTERVAL  4
#define GROUP_SIZE          64
#define BITS                4

// Linear attention (GatedDeltaNet) per-head dimensions
#define LINEAR_KEY_DIM      128
#define LINEAR_VALUE_DIM    128

// Derived linear attention totals
#define LINEAR_TOTAL_KEY    (LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM)
#define LINEAR_TOTAL_VALUE  (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM)
#define LINEAR_CONV_DIM     (LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE)
#define CONV_KERNEL_SIZE    4

// Full attention constants
#define ROPE_THETA          10000000.0f
#define PARTIAL_ROTARY      0.25f
#define ROTARY_DIM          (int)(HEAD_DIM * PARTIAL_ROTARY)  // 64

// Derived layer counts
#define NUM_FULL_ATTN_LAYERS  (NUM_LAYERS / FULL_ATTN_INTERVAL)
#define NUM_LINEAR_LAYERS     (NUM_LAYERS - NUM_FULL_ATTN_LAYERS)

#endif // MODEL_CONFIG_H
