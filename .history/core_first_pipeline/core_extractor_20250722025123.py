"""
Core Extraction Module for TMA Processing

This module handles the extraction of individual tissue cores from TMA images
based on detection results. It preserves all channels for multi-channel images
and creates organized output structures for downstream processing.

Key Features:
- Extract cores with all channels preserved (especially for Orion multi-channel images)
- Organized file structure with metadata
- Quality control and validation
- Batch processing capabilities
- Integration with core detection results
"""

import numpy as np
import cv2
from tifffile import imread, imwrite
from pathlib import Path
import logging
import json
import shutil
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoreExtractionConfig:
    """Configuration for core extraction."""
    
    # Output organization
    output_dir: str = "extracted_cores"
    create_paired_folders: bool = True  # Create paired HE/Orion folders
    
    # File naming
    he_suffix: str = "_HE.tif"
    orion_suffix: str = "_Orion.ome.tif"  # Use .ome.tiff for multi-channel
    metadata_suffix: str = "_metadata.json"
    
    # Extraction parameters
    min_core_size: int = 100  # Minimum core size after extraction
    resize_cores: bool = False  # Whether to resize all cores to same size
    target_size: Tuple[int, int] = (512, 512)  # Target size if resizing
    
    # Quality control
    check_extracted_cores: bool = True
    min_tissue_coverage: float = 0.1  # Minimum fraction of non-background pixels
    
    # Multi-channel handling
    preserve_all_channels: bool = True
    channel_info_file: str = "channel_info.json"
    
    # Data format
    compression: str = "lzw"  # TIFF compression
    save_as_float32: bool = False  # Whether to save as float32 instead of preserving original dtype


class CoreExtractor:
    """Extracts individual cores from TMA images based on detection results."""
    
    def __init__(self, config: CoreExtractionConfig):
        self.config = config
        self.output_path = Path(config.output_dir)
        
    def extract_cores_from_detection(self, image_path: str, detection_results: Dict) -> Dict:
        """
        Extract cores from an image based on detection results.
        
        Args:
            image_path: Path to the source image
            detection_results: Results from CoreDetector
            
        Returns:
            Dictionary with extraction results
        """
        logger.info(f"Extracting {len(detection_results['cores'])} cores from {image_path}")
        
        # Load the full image
        image = imread(image_path)
        image_type = detection_results['image_type']
        
        # Create output directory structure
        image_name = Path(image_path).stem
        image_output_dir = self.output_path / image_name
        image_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract each core
        extraction_results = {
            'source_image': str(image_path),
            'image_type': image_type,
            'output_directory': str(image_output_dir),
            'total_cores_to_extract': len(detection_results['cores']),
            'cores_extracted': [],
            'extraction_errors': [],
            'channel_info': None
        }
        
        # Save channel information for multi-channel images
        if image.ndim == 3 and image.shape[0] > 10:  # Multi-channel
            channel_info = self._create_channel_info(image, image_type)
            extraction_results['channel_info'] = channel_info
            
            # Save channel info to file
            with open(image_output_dir / self.config.channel_info_file, 'w') as f:
                json.dump(channel_info, f, indent=2)
        
        # Extract cores
        for core in detection_results['cores']:
            try:
                core_result = self._extract_single_core(image, core, image_output_dir, image_type)
                extraction_results['cores_extracted'].append(core_result)
                
            except Exception as e:
                error_msg = f"Failed to extract core {core['id']}: {str(e)}"
                logger.error(error_msg)
                extraction_results['extraction_errors'].append(error_msg)
        
        # Save extraction summary
        self._save_extraction_summary(extraction_results, image_output_dir)
        
        logger.info(f"Successfully extracted {len(extraction_results['cores_extracted'])} cores")
        return extraction_results
    
    def _extract_single_core(self, image: np.ndarray, core: Dict, output_dir: Path, image_type: str) -> Dict:
        """Extract a single core from the image."""
        
        core_id = f"core_{core['id']:03d}"
        
        # Get bounding box
        minr, minc, maxr, maxc = core['bbox_padded']
        
        # Extract core region
        if image.ndim == 3 and image.shape[0] <= 100:  # Multi-channel (C, H, W)
            core_image = image[:, minr:maxr, minc:maxc]
        else:  # RGB (H, W, C) or grayscale (H, W)
            core_image = image[minr:maxr, minc:maxc]
        
        # Validate extracted core
        if core_image.size == 0:
            raise ValueError(f"Extracted core {core_id} is empty")
        
        if min(core_image.shape[-2:]) < self.config.min_core_size:
            raise ValueError(f"Extracted core {core_id} too small: {core_image.shape}")
        
        # Resize if requested
        if self.config.resize_cores:
            core_image = self._resize_core(core_image, self.config.target_size)
        
        # Quality check
        if self.config.check_extracted_cores:
            quality_score = self._assess_core_quality(core_image)
            if quality_score < self.config.min_tissue_coverage:
                logger.warning(f"Core {core_id} has low tissue coverage: {quality_score:.2f}")
        else:
            quality_score = None
        
        # Determine output filename based on image type
        if image_type == 'he':
            filename = f"{core_id}{self.config.he_suffix}"
        else:
            filename = f"{core_id}{self.config.orion_suffix}"
        
        core_path = output_dir / filename
        
        # Save core image
        if self.config.save_as_float32 and core_image.dtype != np.float32:
            core_image = core_image.astype(np.float32)
        
        imwrite(str(core_path), core_image, compression=self.config.compression)
        
        # Create core metadata
        core_metadata = {
            'core_id': core_id,
            'source_image_type': image_type,
            'detection_info': core,
            'extracted_shape': list(core_image.shape),
            'extracted_dtype': str(core_image.dtype),
            'file_path': str(core_path),
            'quality_score': quality_score,
            'extraction_bbox': [int(minr), int(minc), int(maxr), int(maxc)]
        }
        
        # Save metadata
        metadata_path = output_dir / f"{core_id}{self.config.metadata_suffix}"
        with open(metadata_path, 'w') as f:
            json.dump(core_metadata, f, indent=2, default=str)
        
        logger.debug(f"Extracted {core_id}: shape {core_image.shape}, saved to {core_path}")
        
        return core_metadata
    
    def _create_channel_info(self, image: np.ndarray, image_type: str) -> Dict:
        """Create channel information for multi-channel images."""
        
        num_channels = image.shape[0] if image.ndim == 3 and image.shape[0] <= 100 else 1
        
        # Common channel mappings
        if image_type == 'orion':
            # Typical Orion channel mapping (this may need adjustment based on your data)
            common_channels = [
                "DAPI", "CD3", "CD20", "CD68", "CD8", "FOXP3", "Ki67", "PanCK", 
                "SMA", "Vimentin", "CD31", "CD45", "PDL1", "CD4", "CD19", "CD56",
                "CD57", "CD163", "CD15", "Blank"
            ]
            
            channel_info = {
                'num_channels': num_channels,
                'channel_format': 'C,H,W' if image.ndim == 3 and image.shape[0] <= 100 else 'H,W,C',
                'channels': []
            }
            
            for i in range(num_channels):
                channel_name = common_channels[i] if i < len(common_channels) else f"Channel_{i}"
                channel_info['channels'].append({
                    'index': i,
                    'name': channel_name,
                    'description': f"Channel {i}" + (f" ({channel_name})" if channel_name != f"Channel_{i}" else "")
                })
        
        else:  # H&E or other
            channel_info = {
                'num_channels': num_channels,
                'channel_format': 'H,W,C' if image.ndim == 3 else 'H,W',
                'channels': []
            }
            
            if num_channels == 3:
                for i, name in enumerate(['Red', 'Green', 'Blue']):
                    channel_info['channels'].append({
                        'index': i,
                        'name': name,
                        'description': f"{name} channel"
                    })
            else:
                for i in range(num_channels):
                    channel_info['channels'].append({
                        'index': i,
                        'name': f"Channel_{i}",
                        'description': f"Channel {i}"
                    })
        
        return channel_info
    
    def _resize_core(self, core_image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize a core image to target size while preserving aspect ratio."""
        
        target_h, target_w = target_size
        
        if core_image.ndim == 3 and core_image.shape[0] <= 100:  # Multi-channel (C, H, W)
            current_h, current_w = core_image.shape[1], core_image.shape[2]
            
            # Calculate scaling factor to fit within target size
            scale_h = target_h / current_h
            scale_w = target_w / current_w
            scale = min(scale_h, scale_w)
            
            new_h = int(current_h * scale)
            new_w = int(current_w * scale)
            
            # Resize each channel
            resized_channels = []
            for c in range(core_image.shape[0]):
                resized_channel = cv2.resize(core_image[c], (new_w, new_h), interpolation=cv2.INTER_AREA)
                resized_channels.append(resized_channel)
            
            resized_image = np.stack(resized_channels, axis=0)
            
            # Pad to exact target size if needed
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            
            if pad_h > 0 or pad_w > 0:
                padding = ((0, 0), (pad_h, target_h - new_h - pad_h), (pad_w, target_w - new_w - pad_w))
                resized_image = np.pad(resized_image, padding, mode='constant', constant_values=0)
        
        else:  # RGB (H, W, C) or grayscale (H, W)
            current_h, current_w = core_image.shape[:2]
            
            scale_h = target_h / current_h
            scale_w = target_w / current_w
            scale = min(scale_h, scale_w)
            
            new_h = int(current_h * scale)
            new_w = int(current_w * scale)
            
            resized_image = cv2.resize(core_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Pad to exact target size if needed
            if core_image.ndim == 3:  # RGB
                pad_h = (target_h - new_h) // 2
                pad_w = (target_w - new_w) // 2
                padding = ((pad_h, target_h - new_h - pad_h), (pad_w, target_w - new_w - pad_w), (0, 0))
            else:  # Grayscale
                pad_h = (target_h - new_h) // 2
                pad_w = (target_w - new_w) // 2
                padding = ((pad_h, target_h - new_h - pad_h), (pad_w, target_w - new_w - pad_w))
            
            if pad_h > 0 or pad_w > 0:
                resized_image = np.pad(resized_image, padding, mode='constant', constant_values=0)
        
        return resized_image
    
    def _assess_core_quality(self, core_image: np.ndarray) -> float:
        """Assess the quality of an extracted core (fraction of non-background pixels)."""
        
        # Use first channel or grayscale for assessment
        if core_image.ndim == 3 and core_image.shape[0] <= 100:  # Multi-channel (C, H, W)
            assessment_image = core_image[0]
        elif core_image.ndim == 3:  # RGB (H, W, C)
            assessment_image = cv2.cvtColor(core_image, cv2.COLOR_RGB2GRAY)
        else:  # Grayscale
            assessment_image = core_image
        
        # Simple thresholding to identify tissue vs background
        threshold = 10  # Adjust based on your data
        tissue_mask = assessment_image > threshold
        tissue_coverage = np.mean(tissue_mask)
        
        return tissue_coverage
    
    def _save_extraction_summary(self, results: Dict, output_dir: Path):
        """Save extraction summary to file."""
        
        summary_file = output_dir / "extraction_summary.json"
        
        # Create a summary without large data structures
        summary = {
            'source_image': results['source_image'],
            'image_type': results['image_type'],
            'total_cores_to_extract': results['total_cores_to_extract'],
            'cores_extracted_count': len(results['cores_extracted']),
            'extraction_errors_count': len(results['extraction_errors']),
            'extraction_errors': results['extraction_errors'],
            'output_directory': results['output_directory'],
            'channel_info': results['channel_info']
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Also create a CSV with core information
        if results['cores_extracted']:
            cores_df = pd.DataFrame([
                {
                    'core_id': core['core_id'],
                    'shape': str(core['extracted_shape']),
                    'dtype': core['extracted_dtype'],
                    'quality_score': core.get('quality_score'),
                    'file_path': Path(core['file_path']).name
                }
                for core in results['cores_extracted']
            ])
            
            cores_df.to_csv(output_dir / "extracted_cores.csv", index=False)
    
    def extract_paired_cores(self, he_detection_results: Dict, orion_detection_results: Dict,
                           he_image_path: str, orion_image_path: str) -> Dict:
        """
        Extract cores from paired H&E and Orion images (for when you have matched detections).
        This is a placeholder for future paired extraction functionality.
        """
        # For now, extract them separately
        he_results = self.extract_cores_from_detection(he_image_path, he_detection_results)
        orion_results = self.extract_cores_from_detection(orion_image_path, orion_detection_results)
        
        return {
            'he_extraction': he_results,
            'orion_extraction': orion_results,
            'paired': False  # Will be True when matching is implemented
        }


def main():
    """Example usage of the CoreExtractor."""
    
    # This would typically be called after core detection
    from core_detector import CoreDetector, CoreDetectionConfig
    
    # Configuration
    detection_config = CoreDetectionConfig()
    extraction_config = CoreExtractionConfig(
        output_dir="extracted_cores",
        preserve_all_channels=True
    )
    
    # Create detector and extractor
    detector = CoreDetector(detection_config)
    extractor = CoreExtractor(extraction_config)
    
    # Example paths
    he_path = "data/raw/TA118-HEraw.ome.tiff"
    orion_path = "data/raw/TA118-Orionraw.ome.tiff"
    
    # Process H&E
    if Path(he_path).exists():
        he_detection = detector.detect_cores(he_path, image_type="he")
        he_extraction = extractor.extract_cores_from_detection(he_path, he_detection)
        print(f"H&E: Extracted {len(he_extraction['cores_extracted'])} cores")
    
    # Process Orion
    if Path(orion_path).exists():
        orion_detection = detector.detect_cores(orion_path, image_type="orion")
        orion_extraction = extractor.extract_cores_from_detection(orion_path, orion_detection)
        print(f"Orion: Extracted {len(orion_extraction['cores_extracted'])} cores")


if __name__ == "__main__":
    main() 