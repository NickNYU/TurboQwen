"""
Lloyd-Max codebook precomputation for optimal scalar quantization of Hadamard-rotated vectors.

For dimension d=256, after Hadamard rotation each coordinate is approximately
N(0, 1/sqrt(d)) = N(0, 1/16), so sigma = 1/16 = 0.0625.

Compute optimal Lloyd-Max centroids for b=1,2,3,4 bits.
Pure numpy implementation (no scipy dependency).
"""

import math
import numpy as np


SIGMA = 1.0 / 16.0  # 0.0625, stddev after Hadamard rotation of d=256 vectors

# ─── Pure-numpy/math Gaussian helpers ─────────────────────────────────────────

def norm_cdf_scalar(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf_scalar(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def truncated_mean(lo, hi, sigma=SIGMA):
    """E[X | lo < X < hi] for X ~ N(0, sigma^2)."""
    a = lo / sigma
    b = hi / sigma
    phi_a = norm_pdf_scalar(a)
    phi_b = norm_pdf_scalar(b)
    Phi_a = norm_cdf_scalar(a)
    Phi_b = norm_cdf_scalar(b)
    denom = Phi_b - Phi_a
    if denom < 1e-15:
        return (lo + hi) / 2.0
    return sigma * (phi_a - phi_b) / denom


def lloyd_max(n_centroids, sigma=SIGMA, n_iters=1000, tol=1e-12):
    """
    1D Lloyd-Max iteration for X ~ N(0, sigma^2).
    Returns (centroids, boundaries) both sorted ascending.
    """
    centroids = np.linspace(-3.0 * sigma, 3.0 * sigma, n_centroids)

    for _ in range(n_iters):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        ext = np.concatenate([[-np.inf], boundaries, [np.inf]])

        new_centroids = np.array([
            truncated_mean(ext[i], ext[i + 1], sigma)
            for i in range(n_centroids)
        ])

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids, boundaries


def compute_mse(centroids, boundaries, sigma=SIGMA):
    """Expected squared quantization error for the given codebook."""
    ext = np.concatenate([[-np.inf], boundaries, [np.inf]])
    mse = 0.0
    for i, c in enumerate(centroids):
        lo, hi = ext[i], ext[i + 1]
        a = lo / sigma if lo != -np.inf else -8.0
        b = hi / sigma if hi != np.inf else  8.0
        prob = norm_cdf_scalar(b) - norm_cdf_scalar(a)
        if prob < 1e-15:
            continue
        ex = truncated_mean(lo, hi, sigma) * prob
        # E[X^2 | lo<X<hi]*P = sigma^2*(phi(a)*a - phi(b)*b + Phi(b) - Phi(a))
        ex2 = sigma ** 2 * (
            norm_pdf_scalar(a) * a - norm_pdf_scalar(b) * b
            + norm_cdf_scalar(b) - norm_cdf_scalar(a)
        )
        mse += ex2 - 2 * c * ex + c ** 2 * prob
    return mse


def format_c_array(name, values):
    vals_str = ", ".join(f"{v:.8f}f" for v in values)
    return f"static const float {name}[] = {{ {vals_str} }};"


def main():
    print(f"// Lloyd-Max codebooks for N(0, sigma^2), sigma = 1/16 = {SIGMA}")
    print(f"// Dimension d=256 (Hadamard-rotated KV cache vectors)")
    print()

    for bits in [1, 2, 3, 4]:
        n = 2 ** bits
        centroids, boundaries = lloyd_max(n, sigma=SIGMA)

        print(f"// b={bits} ({bits}-bit, {n} centroids) for N(0, 1/16)")
        print(format_c_array(f"TQ_CODEBOOK_{bits}BIT", centroids))
        if len(boundaries) > 0:
            print(format_c_array(f"TQ_BOUNDS_{bits}BIT", boundaries))
        else:
            print(f"static const float TQ_BOUNDS_{bits}BIT[] = {{ 0.0f }};")
        mse = compute_mse(centroids, boundaries, sigma=SIGMA)
        print(f"// Centroids (units of sigma): {[round(c/SIGMA, 4) for c in centroids]}")
        print(f"// Lloyd-Max MSE: {mse:.8f}  (normalized MSE = {mse/SIGMA**2:.6f})")
        print()


if __name__ == "__main__":
    main()
