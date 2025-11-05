"""
Core-First TMA Processing Pipeline

A comprehensive pipeline for processing Tissue Microarray (TMA) images using a 
core-first approach instead of whole-slide registration. This approach is more
efficient, reliable, and scientifically sound for TMA analysis.

Modules:
- core_detector: Detects tissue cores in TMA images using multiple algorithms
- core_matcher: Matches cores spatially between H&E and multiplex images  
- core_extractor: Extracts individual cores while preserving all channels
- core_first_pipeline: Main orchestrator that combines all components

Example Usage:
    from core_first_pipeline import CoreFirstPipeline, CoreFirstPipelineConfig
    
    config = CoreFirstPipelineConfig(
        he_image_path="path/to/he_image.tiff",
        orion_image_path="path/to/orion_image.tiff",
        output_dir="output_directory"
    )
    
    pipeline = CoreFirstPipeline(config)
    results = pipeline.run()
"""

from .core_detector import CoreDetector, CoreDetectionConfig
from .core_matcher import CoreMatcher, CoreMatchingConfig
from .core_extractor import CoreExtractor, CoreExtractionConfig
from .core_first_pipeline import CoreFirstPipeline, CoreFirstPipelineConfig

__version__ = "1.0.0"
__author__ = "Core-First Pipeline Team"

__all__ = [
    "CoreDetector", "CoreDetectionConfig",
    "CoreMatcher", "CoreMatchingConfig", 
    "CoreExtractor", "CoreExtractionConfig",
    "CoreFirstPipeline", "CoreFirstPipelineConfig"
] 