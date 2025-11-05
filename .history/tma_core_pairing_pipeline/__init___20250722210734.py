"""
TMA Core Pairing Pipeline

Robust dual-modality core detection and registration pipeline for H&E and Orion TMA images.
"""

__version__ = "1.0.0"
__author__ = "Stanford Biomedical Data Science Lab"

from .dual_modality_core_detector import DualModalityCoreDetector
from .matched_core_registration_pipeline import MatchedCoreRegistrationPipeline

__all__ = [
    "DualModalityCoreDetector",
    "MatchedCoreRegistrationPipeline"
] 