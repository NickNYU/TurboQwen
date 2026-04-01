"""
TurboQuant quality benchmark: float32 vs QJL vs TurboQuant at various bit-widths.

Parameters: HEAD_DIM=256, NUM_KV_HEADS=2, SEQ_LEN=512
"""

import math
import numpy as np

# ─── Configuration ─────────────────────────────────────────────────────────────
HEAD_DIM = 256
NUM_KV_HEADS = 2
SEQ_LEN = 512
N_PAIRS = 1000
SIGMA = 1.0 / np.sqrt(HEAD_DIM)  # 1/16 for d=256
RNG_SEED = 42

# ─── Walsh-Hadamard Transform ───────────────────────────────────────────────────

def walsh_hadamard_transform(x):
    """In-place Walsh-Hadamard transform, normalized by 1/sqrt(dim)."""
    dim = len(x)
    assert dim & (dim - 1) == 0, "dim must be power of 2"
    h = x.copy().astype(np.float64)
    stride = 1
    while stride < dim:
        for i in range(dim // 2):
            lo = (i // stride) * (stride * 2) + (i % stride)
            hi = lo + stride
            a, b = h[lo], h[hi]
            h[lo] = a + b
            h[hi] = a - b
        stride *= 2
    return h / np.sqrt(dim)


def fast_wht(x):
    """Vectorized WHT using recursive doubling (much faster for benchmarking)."""
    dim = len(x)
    h = x.copy().astype(np.float64)
    length = 1
    while length < dim:
        h = h.reshape(-1, 2 * length)
        lo = h[:, :length].copy()
        hi = h[:, length:].copy()
        h[:, :length] = lo + hi
        h[:, length:] = lo - hi
        length *= 2
    return h.ravel() / np.sqrt(dim)


# ─── Codebook (Lloyd-Max, precomputed for N(0, 1/16)) ──────────────────────────

def make_uniform_codebook(bits, sigma=SIGMA):
    """Uniform codebook on [-3sigma, 3sigma] as baseline."""
    n = 2 ** bits
    return np.linspace(-3 * sigma, 3 * sigma, n)


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def lloyd_max_codebook(bits, sigma=SIGMA, n_iters=500):
    """Lloyd-Max 1D k-means on N(0, sigma^2). Pure numpy."""

    def trunc_mean(lo, hi):
        a = lo / sigma if lo != -np.inf else -8.0
        b = hi / sigma if hi != np.inf else  8.0
        phi_a, phi_b = _norm_pdf(a), _norm_pdf(b)
        denom = _norm_cdf(b) - _norm_cdf(a)
        if denom < 1e-15:
            return (lo + hi) / 2.0
        return sigma * (phi_a - phi_b) / denom

    n = 2 ** bits
    centroids = np.linspace(-3 * sigma, 3 * sigma, n)
    for _ in range(n_iters):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        ext = np.concatenate([[-np.inf], boundaries, [np.inf]])
        new_c = np.array([trunc_mean(ext[i], ext[i + 1]) for i in range(n)])
        if np.max(np.abs(new_c - centroids)) < 1e-12:
            centroids = new_c
            break
        centroids = new_c
    return centroids


# ─── QJL ───────────────────────────────────────────────────────────────────────

def qjl_encode(k, R):
    """QJL: sign(R @ k), returns sign bits as int8 (0/1)."""
    proj = R @ k
    return (proj >= 0).astype(np.int8)


def qjl_dot_estimate(q_signs, k_signs, dim):
    """Estimate dot product from QJL signs: (d - 2*hamming) / d * sqrt(pi/2) * norms."""
    hamming = np.sum(q_signs != k_signs)
    return (dim - 2 * hamming) / dim


# ─── TurboQuant ────────────────────────────────────────────────────────────────

def turboquant_encode(k, codebook, R):
    """TurboQuant: Hadamard + MSE quantize + QJL on residual."""
    rotated = fast_wht(k)
    indices = np.argmin(np.abs(rotated[:, None] - codebook[None, :]), axis=1)
    dequant = codebook[indices]
    residual = rotated - dequant
    r_norm = np.linalg.norm(residual)
    qjl_signs = qjl_encode(residual, R)
    return rotated, indices, dequant, residual, r_norm, qjl_signs


def turboquant_score(rotated_q, q_proj_signs, mse_dequant, qjl_k_signs, r_norm, dim):
    """Reconstruct attention score from TQ-compressed K."""
    mse_term = np.dot(rotated_q, mse_dequant)
    qjl_term = np.sqrt(np.pi / 2) * r_norm * qjl_dot_estimate(q_proj_signs, qjl_k_signs, dim)
    return mse_term + qjl_term


# ─── Attention accuracy ─────────────────────────────────────────────────────────

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def kl_div(p, q, eps=1e-12):
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return np.sum(p * np.log(p / q))


# ─── Main benchmark ─────────────────────────────────────────────────────────────

def run_benchmark():
    rng = np.random.default_rng(RNG_SEED)

    # Random projection matrix for QJL (fixed, shared)
    R = rng.standard_normal((HEAD_DIM, HEAD_DIM))

    # Build codebooks
    codebooks = {}
    print("Building Lloyd-Max codebooks...")
    for bits in [2, 3, 4]:
        codebooks[bits] = lloyd_max_codebook(bits)
        print(f"  {bits}-bit: {len(codebooks[bits])} centroids, "
              f"range [{codebooks[bits][0]:.5f}, {codebooks[bits][-1]:.5f}]")

    # ── Per-pair error metrics ──────────────────────────────────────────────────
    methods = ["QJL-1bit"] + [f"TQ-{b}bit" for b in [2, 3, 4]]
    errors = {m: [] for m in methods}

    # ── Attention distribution accuracy (seq_len=512 keys per query) ───────────
    attn_kl = {m: [] for m in methods}

    print(f"\nRunning {N_PAIRS} random Q/K pairs (HEAD_DIM={HEAD_DIM})...")

    for trial in range(N_PAIRS):
        q = rng.standard_normal(HEAD_DIM) * SIGMA
        k = rng.standard_normal(HEAD_DIM) * SIGMA
        true_score = np.dot(q, k)

        # QJL
        q_signs = qjl_encode(q, R)
        k_signs = qjl_encode(k, R)
        qjl_est = (np.sqrt(np.pi / 2)
                   * np.linalg.norm(q) * np.linalg.norm(k)
                   * qjl_dot_estimate(q_signs, k_signs, HEAD_DIM))
        errors["QJL-1bit"].append(abs(qjl_est - true_score))

        # TurboQuant variants
        rotated_q = fast_wht(q)
        q_proj_signs = qjl_encode(rotated_q, R)

        for bits in [2, 3, 4]:
            cb = codebooks[bits]
            _, _, dequant_k, _, r_norm, qjl_k_signs = turboquant_encode(k, cb, R)
            tq_est = turboquant_score(rotated_q, q_proj_signs, dequant_k,
                                      qjl_k_signs, r_norm, HEAD_DIM)
            errors[f"TQ-{bits}bit"].append(abs(tq_est - true_score))

    # ── Attention accuracy over SEQ_LEN keys ───────────────────────────────────
    print(f"Running attention accuracy test (seq_len={SEQ_LEN})...")
    for _ in range(50):  # 50 attention heads worth of tests
        q = rng.standard_normal(HEAD_DIM) * SIGMA
        keys = rng.standard_normal((SEQ_LEN, HEAD_DIM)) * SIGMA

        true_scores = keys @ q
        true_attn = softmax(true_scores / np.sqrt(HEAD_DIM))

        rotated_q = fast_wht(q)
        q_proj_signs = qjl_encode(rotated_q, R)
        q_signs_raw = qjl_encode(q, R)

        for method in methods:
            est_scores = np.zeros(SEQ_LEN)
            for j, k_vec in enumerate(keys):
                if method == "QJL-1bit":
                    k_signs_raw = qjl_encode(k_vec, R)
                    est_scores[j] = (np.sqrt(np.pi / 2)
                                     * np.linalg.norm(q) * np.linalg.norm(k_vec)
                                     * qjl_dot_estimate(q_signs_raw, k_signs_raw, HEAD_DIM))
                else:
                    bits = int(method.split("-")[1].replace("bit", ""))
                    cb = codebooks[bits]
                    _, _, dequant_k, _, r_norm, qjl_k_signs = turboquant_encode(k_vec, cb, R)
                    est_scores[j] = turboquant_score(rotated_q, q_proj_signs, dequant_k,
                                                     qjl_k_signs, r_norm, HEAD_DIM)
            est_attn = softmax(est_scores / np.sqrt(HEAD_DIM))
            attn_kl[method].append(kl_div(true_attn, est_attn))

    # ── Print results ───────────────────────────────────────────────────────────
    print()
    print("=" * 75)
    print(f"{'Method':<12} {'Mean |err|':>12} {'Max |err|':>12} "
          f"{'Rel MSE':>12} {'Bias':>10} {'KL(attn)':>10}")
    print("-" * 75)

    true_var = SIGMA ** 2  # variance of dot product ~ d * sigma^4

    for method in ["Float32"] + methods:
        if method == "Float32":
            print(f"{'Float32':<12} {'0.000000':>12} {'0.000000':>12} "
                  f"{'0.000000':>12} {'0.000000':>10} {'0.000000':>10}  (ground truth)")
            continue

        errs = np.array(errors[method])
        kls = np.array(attn_kl[method])

        # Signed errors for bias
        rng2 = np.random.default_rng(RNG_SEED)
        R2 = rng2.standard_normal((HEAD_DIM, HEAD_DIM))
        signed = []
        for trial in range(N_PAIRS):
            q = rng2.standard_normal(HEAD_DIM) * SIGMA
            k = rng2.standard_normal(HEAD_DIM) * SIGMA
            true_score = np.dot(q, k)
            if method == "QJL-1bit":
                q_s = qjl_encode(q, R2)
                k_s = qjl_encode(k, R2)
                est = (np.sqrt(np.pi / 2)
                       * np.linalg.norm(q) * np.linalg.norm(k)
                       * qjl_dot_estimate(q_s, k_s, HEAD_DIM))
            else:
                bits = int(method.split("-")[1].replace("bit", ""))
                cb = codebooks[bits]
                rq = fast_wht(q)
                qps = qjl_encode(rq, R2)
                _, _, dq, _, rn, qjl_ks = turboquant_encode(k, cb, R2)
                est = turboquant_score(rq, qps, dq, qjl_ks, rn, HEAD_DIM)
            signed.append(est - true_score)

        bias = np.mean(signed)
        mse = np.mean(np.array(signed) ** 2)
        rel_mse = mse / max(true_var * HEAD_DIM, 1e-15)

        print(f"{method:<12} {np.mean(errs):>12.6f} {np.max(errs):>12.6f} "
              f"{rel_mse:>12.6f} {bias:>10.6f} {np.mean(kls):>10.6f}")

    print("=" * 75)
    print(f"\nNote: HEAD_DIM={HEAD_DIM}, N_PAIRS={N_PAIRS}, SEQ_LEN={SEQ_LEN}, "
          f"sigma=1/sqrt({HEAD_DIM})={SIGMA:.4f}")


if __name__ == "__main__":
    run_benchmark()
