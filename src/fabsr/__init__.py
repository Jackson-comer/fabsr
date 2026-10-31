"""fabsr: train ternary language models by flipping weights stored as 2-bit codes. No latent weights; the update runs inside backward.

See fabsr.api for the public functions. Nothing heavy is imported until you use it.
"""
from .api import Fabsr, load_codes, load_falcon_e, rate_schedule, save_codes, ternary_layers

__version__ = "0.1.0"
__all__ = ["Fabsr", "load_falcon_e", "ternary_layers", "save_codes", "load_codes", "rate_schedule", "__version__"]
