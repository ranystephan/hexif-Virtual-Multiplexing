"""
Configuration file for TMA Core Pairing Pipeline

This file contains optimized parameter sets for different scenarios and use cases.
Modify these parameters based on your specific TMA images and requirements.
"""

from dataclasses import dataclass
from typing import Optional
from .dual_modality_core_detector import CoreDetectionConfig

# Try to import registration config if available
try:
    from registration_pipeline import RegistrationConfig
    REGISTRATION_AVAILABLE = True
except ImportError:
    REGISTRATION_AVAILABLE = False


@dataclass 
class PipelinePresets:
    """Predefined parameter sets for different scenarios."""
    
    @staticmethod
    def get_default_detection_config(output_dir: str = "./tma_pipeline_output") -> CoreDetectionConfig:
        """Default configuration optimized for typical TMA images."""
        return CoreDetectionConfig(
            # SpaceC parameters
            downscale_factor=64,
            padding=50,
            
            # H&E detection (optimized for standard H&E staining)
            he_lower_cutoff=0.15,
            he_upper_cutoff=0.25,
            
            # Orion detection (optimized for DAPI channel)
            orion_lower_cutoff=0.10,
            orion_upper_cutoff=0.20,
            dapi_channel=0,
            
            # Core matching (balanced for accuracy and coverage)
            max_match_distance=500.0,
            min_size_ratio=0.4,
            max_size_ratio=2.5,
            min_circularity=0.2,
            
            # Quality control
            min_core_area=10000,
            max_core_area=1000000,
            
            # Processing
            temp_dir=output_dir,
            save_debug_images=True
        )
    
    @staticmethod
    def get_high_sensitivity_config(output_dir: str = "./tma_pipeline_output") -> CoreDetectionConfig:
        """Configuration for detecting more cores (may include some false positives)."""
        return CoreDetectionConfig(
            # More aggressive detection
            downscale_factor=32,  # Higher resolution
            padding=50,
            
            # Broader H&E detection range
            he_lower_cutoff=0.10,
            he_upper_cutoff=0.30,
            
            # Broader Orion detection range  
            orion_lower_cutoff=0.05,
            orion_upper_cutoff=0.25,
            dapi_channel=0,
            
            # More lenient matching
            max_match_distance=750.0,
            min_size_ratio=0.3,
            max_size_ratio=3.0,
            min_circularity=0.1,
            
            # Broader size range
            min_core_area=5000,
            max_core_area=1500000,
            
            temp_dir=output_dir,
            save_debug_images=True
        )
    
    @staticmethod
    def get_high_precision_config(output_dir: str = "./tma_pipeline_output") -> CoreDetectionConfig:
        """Configuration for high precision (fewer cores but higher quality matches)."""
        return CoreDetectionConfig(
            # Standard resolution
            downscale_factor=64,
            padding=50,
            
            # Stricter H&E detection
            he_lower_cutoff=0.18,
            he_upper_cutoff=0.22,
            
            # Stricter Orion detection
            orion_lower_cutoff=0.12,
            orion_upper_cutoff=0.18,
            dapi_channel=0,
            
            # Strict matching criteria
            max_match_distance=300.0,
            min_size_ratio=0.5,
            max_size_ratio=2.0,
            min_circularity=0.3,
            
            # Standard size range
            min_core_area=15000,
            max_core_area=800000,
            
            temp_dir=output_dir,
            save_debug_images=True
        )
    
    @staticmethod
    def get_fast_config(output_dir: str = "./tma_pipeline_output") -> CoreDetectionConfig:
        """Configuration optimized for speed (lower resolution, faster processing)."""
        return CoreDetectionConfig(
            # Heavy downscaling for speed
            downscale_factor=128,
            padding=25,
            
            # Standard detection ranges
            he_lower_cutoff=0.15,
            he_upper_cutoff=0.25,
            orion_lower_cutoff=0.10,
            orion_upper_cutoff=0.20,
            dapi_channel=0,
            
            # Moderate matching
            max_match_distance=600.0,
            min_size_ratio=0.4,
            max_size_ratio=2.5,
            min_circularity=0.2,
            
            # Standard quality control
            min_core_area=8000,
            max_core_area=1200000,
            
            temp_dir=output_dir,
            save_debug_images=False  # Skip debug images for speed
        )
    
    @staticmethod
    def get_registration_config(output_dir: str = "./tma_pipeline_output",
                              num_workers: int = 4) -> Optional['RegistrationConfig']:
        """Get optimized VALIS registration configuration if available."""
        if not REGISTRATION_AVAILABLE:
            return None
            
        return RegistrationConfig(
            input_dir=f"{output_dir}/extracted_cores",
            output_dir=f"{output_dir}/registration_output",
            he_suffix="_HE.tif",
            orion_suffix="_Orion.tif",
            
            # VALIS parameters optimized for TMA cores
            max_processed_image_dim_px=1024,        # Good balance of speed/accuracy
            max_non_rigid_registration_dim_px=1500, # Higher res for non-rigid
            reference_img="he",                     # Use H&E as reference
            
            # Training dataset parameters
            patch_size=256,                         # Standard patch size
            stride=256,                            # Non-overlapping patches
            min_background_threshold=10,
            
            # Quality control thresholds
            min_ssim_threshold=0.3,                # Reasonable SSIM threshold
            min_ncc_threshold=0.2,                 # Moderate NCC threshold
            min_mi_threshold=0.5,                  # Mutual information threshold
            
            # Processing parameters
            num_workers=num_workers,               # Parallel processing
            compression="jp2k",                    # Good compression
            compression_quality=95,                # High quality
            
            # Error handling
            skip_failed_registrations=True,       # Continue on failures
            max_failures_before_stop=50,          # Stop if too many failures
            
            # Output formats
            save_ome_tiff=True,
            save_npy_pairs=True,
            save_quality_plots=True
        )


class ParameterTuningGuide:
    """Guidelines for parameter adjustment based on common issues."""
    
    @staticmethod
    def print_tuning_guide():
        """Print comprehensive parameter tuning guide."""
        
        guide = """
        
🔧 TMA CORE PAIRING PARAMETER TUNING GUIDE
═══════════════════════════════════════════

CORE DETECTION ISSUES
─────────────────────

🔴 Too few H&E cores detected:
   • Decrease he_lower_cutoff (0.15 → 0.10)
   • Increase he_upper_cutoff (0.25 → 0.30)
   • Reduce downscale_factor (64 → 32) for more detail
   • Check H&E image contrast and quality

🔵 Too few Orion cores detected:
   • Decrease orion_lower_cutoff (0.10 → 0.05)
   • Increase orion_upper_cutoff (0.20 → 0.25)
   • Verify DAPI channel index (usually 0)
   • Check if Orion image has sufficient contrast

🟡 Too many false positive cores:
   • Increase he_lower_cutoff (0.15 → 0.20)
   • Decrease he_upper_cutoff (0.25 → 0.22)
   • Increase min_core_area (10000 → 20000)
   • Increase min_circularity (0.2 → 0.3)

CORE MATCHING ISSUES
────────────────────

🟠 Low matching rate (<70%):
   • Increase max_match_distance (500 → 750)
   • Relax size ratio bounds (0.4-2.5 → 0.3-3.0)
   • Decrease min_circularity (0.2 → 0.1)
   • Check for image orientation differences

🟣 High match distances (>300 pixels):
   • Verify images are from same TMA
   • Check for image flips or rotations
   • Consider global pre-alignment
   • Reduce max_match_distance to filter bad matches

PERFORMANCE ISSUES
─────────────────

⚡ Processing too slow:
   • Increase downscale_factor (64 → 128)
   • Reduce padding (50 → 25)
   • Set save_debug_images=False
   • Use fewer parallel workers if memory limited

💾 Memory issues:
   • Increase downscale_factor
   • Process images sequentially (num_workers=1)
   • Clear variables between processing steps
   • Check available RAM vs image sizes

PARAMETER RANGES
───────────────

Detection Cutoffs:
• he_lower_cutoff:    0.05 - 0.30  (default: 0.15)
• he_upper_cutoff:    0.15 - 0.40  (default: 0.25)
• orion_lower_cutoff: 0.02 - 0.25  (default: 0.10)  
• orion_upper_cutoff: 0.08 - 0.35  (default: 0.20)

Processing Parameters:
• downscale_factor:   16 - 256     (default: 64)
• padding:            10 - 100     (default: 50)

Matching Parameters:
• max_match_distance: 100 - 1500   (default: 500)
• min_size_ratio:     0.2 - 0.8    (default: 0.4)
• max_size_ratio:     1.5 - 5.0    (default: 2.5)
• min_circularity:    0.05 - 0.6   (default: 0.2)

USAGE WORKFLOW
─────────────

1. 🚀 Start with default configuration
2. 📊 Run detection and examine results  
3. 🔧 Adjust parameters based on issues found
4. 🔄 Re-run detection with new parameters
5. ✅ Proceed to full pipeline when satisfied

PRESET CONFIGURATIONS
────────────────────

• Default: Balanced performance for typical TMAs
• High Sensitivity: Detect more cores (may have false positives)
• High Precision: Fewer cores but higher quality matches
• Fast: Optimized for speed over accuracy

        """
        
        print(guide)
    
    @staticmethod
    def suggest_parameters_for_results(he_detected: int, orion_detected: int, 
                                     matched: int, matching_rate: float,
                                     mean_distance: float, current_config: CoreDetectionConfig) -> CoreDetectionConfig:
        """
        Suggest parameter adjustments based on detection results.
        
        Args:
            he_detected: Number of H&E cores detected
            orion_detected: Number of Orion cores detected  
            matched: Number of matched cores
            matching_rate: Fraction of cores successfully matched
            mean_distance: Average distance between matched cores
            current_config: Current configuration
            
        Returns:
            Updated configuration with suggested parameter changes
        """
        
        print("🔧 AUTOMATED PARAMETER SUGGESTIONS")
        print("=" * 50)
        
        # Create a copy of current config
        new_config = CoreDetectionConfig(
            downscale_factor=current_config.downscale_factor,
            padding=current_config.padding,
            he_lower_cutoff=current_config.he_lower_cutoff,
            he_upper_cutoff=current_config.he_upper_cutoff,
            orion_lower_cutoff=current_config.orion_lower_cutoff,
            orion_upper_cutoff=current_config.orion_upper_cutoff,
            dapi_channel=current_config.dapi_channel,
            max_match_distance=current_config.max_match_distance,
            min_size_ratio=current_config.min_size_ratio,
            max_size_ratio=current_config.max_size_ratio,
            min_circularity=current_config.min_circularity,
            min_core_area=current_config.min_core_area,
            max_core_area=current_config.max_core_area,
            temp_dir=current_config.temp_dir,
            save_debug_images=current_config.save_debug_images
        )
        
        suggestions = []
        
        # Analyze H&E detection
        if he_detected < 200:
            new_config.he_lower_cutoff = max(0.05, current_config.he_lower_cutoff - 0.05)
            new_config.he_upper_cutoff = min(0.40, current_config.he_upper_cutoff + 0.05)
            suggestions.append(f"🔴 H&E detection: Adjusted cutoffs to {new_config.he_lower_cutoff:.3f}-{new_config.he_upper_cutoff:.3f}")
        
        # Analyze Orion detection  
        if orion_detected < 200:
            new_config.orion_lower_cutoff = max(0.02, current_config.orion_lower_cutoff - 0.03)
            new_config.orion_upper_cutoff = min(0.35, current_config.orion_upper_cutoff + 0.03)
            suggestions.append(f"🔵 Orion detection: Adjusted cutoffs to {new_config.orion_lower_cutoff:.3f}-{new_config.orion_upper_cutoff:.3f}")
        
        # Analyze matching performance
        if matching_rate < 0.7:
            new_config.max_match_distance = min(1000, current_config.max_match_distance * 1.5)
            new_config.min_size_ratio = max(0.2, current_config.min_size_ratio - 0.1)
            new_config.max_size_ratio = min(4.0, current_config.max_size_ratio + 0.5)
            suggestions.append(f"🟡 Matching: Relaxed distance to {new_config.max_match_distance:.0f}, ratios to {new_config.min_size_ratio:.1f}-{new_config.max_size_ratio:.1f}")
        
        # Analyze spatial accuracy
        if mean_distance > 400:
            new_config.max_match_distance = max(200, current_config.max_match_distance * 0.8)
            suggestions.append(f"🟠 Distance: Reduced max distance to {new_config.max_match_distance:.0f} to filter poor matches")
        
        if suggestions:
            print("Suggested parameter changes:")
            for suggestion in suggestions:
                print(f"  • {suggestion}")
            print(f"\n✅ Generated updated configuration")
        else:
            print("✅ Current parameters appear well-tuned!")
        
        return new_config


# Export commonly used configurations
DEFAULT_CONFIG = PipelinePresets.get_default_detection_config
HIGH_SENSITIVITY_CONFIG = PipelinePresets.get_high_sensitivity_config  
HIGH_PRECISION_CONFIG = PipelinePresets.get_high_precision_config
FAST_CONFIG = PipelinePresets.get_fast_config
REGISTRATION_CONFIG = PipelinePresets.get_registration_config

# Parameter tuning utilities
TUNING_GUIDE = ParameterTuningGuide() 