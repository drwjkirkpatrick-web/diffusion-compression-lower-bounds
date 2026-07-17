# diffusion-compression-lower-bounds

## Information-Theoretic Lower Bounds for Generative/Diffusion Compression with Learned Priors

A research package deriving the theoretical limits of diffusion-based image compression — answering the question: **what is the minimum bitrate when the decoder has a pretrained generative model?**

---

### The Problem

Diffusion-based codecs like DiffC, PerCo, and DiffEIC achieve remarkable compression at ultra-low bitrates (10⁻³ to 0.1 bpp) by using a pretrained diffusion model as decoder side information. But there are **no information-theoretic lower bounds** on what's achievable in this regime. We don't know if these methods are close to optimal or far from it.

### Key Results

#### Theorem 1: Diffusion Compression Lower Bound

For a Gaussian source with a learned prior of quality δ = D_KL(p_X || p_prior):

$$R \geq R(D) - I_{\text{credit}}(\delta) + \Delta_{\text{perception}}(D, P)$$

where:
- **R(D)** is the classical Shannon rate-distortion bound
- **I_credit(δ) = min(δ/ln2, R(D))** is the side information credit from the prior
- **Δ_perception(D, P)** is the RDP perception premium

#### Theorem 2: Ultra-Low Bitrate Distortion Floor

$$D_{\min}(R, \delta) = \sigma^2 \cdot e^{-2\delta} \cdot 2^{-2R}$$

The **distortion floor** at zero rate is D_floor = σ²·e^(-2δ) — the prior quality determines the best achievable quality with no bits at all.

#### Corollary: Prior as Bit Budget

A prior with KL divergence δ provides an effective "head start" of δ/ln(2) bits per sample. The prior reduces the effective source variance from σ² to σ²·e^(-2δ).

### Installation

```bash
git clone https://github.com/drwjkirkpatrick-web/diffusion-compression-lower-bounds.git
cd diffusion-compression-lower-bounds
pip install -e ".[test]"
```

### Usage

```python
from diffusion_compression_lower_bounds import (
    diffusion_compression_lower_bound,
    ultra_low_bitrate_analysis,
    rate_prior_tradeoff,
)

# Lower bound on rate with a prior of quality δ=0.5
bounds = diffusion_compression_lower_bound(
    sigma_sq=4.0, D=1.0, P=float('inf'), kl_divergence=0.5
)
print(f"Lower bound: {bounds.lower_bound:.4f} bits")
print(f"Classical R(D): {bounds.classical_RD:.4f} bits")
print(f"Prior credit: {bounds.side_info_credit:.4f} bits")
print(f"Distortion floor: {bounds.effective_variance:.4f}")

# Ultra-low bitrate analysis
analysis = ultra_low_bitrate_analysis(sigma_sq=4.0, kl_divergence=0.5)
print(f"Distortion floor: {analysis['distortion_floor']:.4f}")
```

### Testing

```bash
python -m pytest tests/ -v
```

### References

- Wyner, A. D. & Ziv, J. (1976). "The rate-distortion function for source coding with side information at the decoder."
- Theis, L. & Ahmed, A. (2022). "Algorithms for the communication of samples." ICML.
- Blau, Y. & Michaeli, T. (2019). "Rethinking lossy compression: The RDP tradeoff." ICML.

### License

MIT