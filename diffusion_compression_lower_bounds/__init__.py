"""
Diffusion Compression Lower Bounds — Information-Theoretic Lower Bounds
for Generative/Diffusion-Based Image Compression.

A research package deriving lower bounds on the achievable compression rate
when the decoder has access to a pretrained generative model (diffusion prior)
as side information.
"""

from diffusion_compression_lower_bounds.core import (
    PriorQuality,
    gaussian_prior_quality,
    side_information_credit,
    DiffusionCompressionBounds,
    diffusion_compression_lower_bound,
    rate_prior_tradeoff,
    ultra_low_bitrate_analysis,
    verify_side_information_credit,
    verify_ultra_low_bitrate_formula,
)

__version__ = "0.1.0"
__author__ = "Walker Kirkpatrick"
__license__ = "MIT"