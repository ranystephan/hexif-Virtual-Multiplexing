"""
Core-First TMA Processing Pipeline

This is the main orchestrator for the core-first approach to TMA processing.
Instead of registering whole slide images, this pipeline:

1. Detects cores in both H&E and Orion images independently
2. Matches cores spatially between the two stains  
3. Extracts matched core pairs while preserving all Orion channels
4. Provides organized output for downstream analysis

This approach is much more efficient and reliable than whole-slide registration
for TMA analysis.

Key Features:
- Independent core detection in each stain
- Spatial matching between stains
- Channel preservation for multi-channel images
- Quality control and validation
- Comprehensive reporting and visualization
- Integration with existing ROSIE workflows
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
import shutil
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd

# Import our modules
from .core_detector import CoreDetector, CoreDetectionConfig
from .core_matcher import CoreMatcher, CoreMatchingConfig  
from .core_extractor import CoreExtractor, CoreExtractionConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoreFirstPipelineConfig:
    """Configuration for the core-first pipeline."""
    
    # Input paths
    he_image_path: str
    orion_image_path: str
    output_dir: str = "core_first_output"
    
    # Processing options
    create_visualizations: bool = True
    save_intermediate_results: bool = True
    create_paired_cores: bool = True
    
    # Quality control
    min_cores_required: int = 5  # Minimum cores needed for successful processing
    max_processing_time: float = 3600  # Maximum processing time in seconds
    
    # Component configurations (can be overridden)
    detection_config: Optional[CoreDetectionConfig] = None
    matching_config: Optional[CoreMatchingConfig] = None
    extraction_config: Optional[CoreExtractionConfig] = None


class CoreFirstPipeline:
    """Main pipeline orchestrator for core-first TMA processing."""
    
    def __init__(self, config: CoreFirstPipelineConfig):
        self.config = config
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create component configurations with defaults if not provided
        self.detection_config = config.detection_config or CoreDetectionConfig(
            detection_method="hybrid",
            min_core_area=30000,
            max_core_area=800000,
            min_circularity=0.25,
            create_visualizations=config.create_visualizations
        )
        
        self.matching_config = config.matching_config or CoreMatchingConfig(
            matching_method="hungarian",
            max_distance_threshold=300.0,
            min_match_confidence=0.4,
            save_visualizations=config.create_visualizations
        )
        
        self.extraction_config = config.extraction_config or CoreExtractionConfig(
            output_dir=str(self.output_path / "extracted_cores"),
            preserve_all_channels=True,
            create_paired_folders=config.create_paired_cores,
            check_extracted_cores=True
        )
        
        # Initialize components
        self.detector = CoreDetector(self.detection_config)
        self.matcher = CoreMatcher(self.matching_config)
        self.extractor = CoreExtractor(self.extraction_config)
        
        # Create output subdirectories
        self.detection_output_dir = self.output_path / "detection_results"
        self.matching_output_dir = self.output_path / "matching_results"  
        self.visualization_output_dir = self.output_path / "visualizations"
        
        for dir_path in [self.detection_output_dir, self.matching_output_dir, self.visualization_output_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def run(self) -> Dict:
        """
        Run the complete core-first processing pipeline.
        
        Returns:
            Dictionary with comprehensive results
        """
        import time
        start_time = time.time()
        
        logger.info("Starting Core-First TMA Processing Pipeline")
        logger.info(f"H&E Image: {self.config.he_image_path}")
        logger.info(f"Orion Image: {self.config.orion_image_path}")
        logger.info(f"Output Directory: {self.config.output_dir}")
        
        # Initialize results structure
        results = {
            'pipeline_config': self.config.__dict__,
            'start_time': start_time,
            'success': False,
            'error': None,
            'processing_stages': {}
        }
        
        try:
            # Stage 1: Core Detection
            logger.info("\n=== STAGE 1: CORE DETECTION ===")
            detection_results = self._run_core_detection()
            results['processing_stages']['detection'] = detection_results
            
            if not detection_results['success']:
                results['error'] = f"Core detection failed: {detection_results.get('error', 'Unknown error')}"
                return results
            
            # Stage 2: Core Matching
            logger.info("\n=== STAGE 2: CORE MATCHING ===")
            matching_results = self._run_core_matching(detection_results)
            results['processing_stages']['matching'] = matching_results
            
            if not matching_results['success']:
                results['error'] = f"Core matching failed: {matching_results.get('error', 'Unknown error')}"
                return results
            
            # Stage 3: Core Extraction
            logger.info("\n=== STAGE 3: CORE EXTRACTION ===")
            extraction_results = self._run_core_extraction(detection_results, matching_results)
            results['processing_stages']['extraction'] = extraction_results
            
            if not extraction_results['success']:
                results['error'] = f"Core extraction failed: {extraction_results.get('error', 'Unknown error')}"
                return results
            
            # Stage 4: Create Visualizations
            if self.config.create_visualizations:
                logger.info("\n=== STAGE 4: VISUALIZATIONS ===")
                visualization_results = self._create_visualizations(detection_results, matching_results)
                results['processing_stages']['visualizations'] = visualization_results
            
            # Stage 5: Generate Final Report
            logger.info("\n=== STAGE 5: FINAL REPORT ===")
            report_results = self._generate_final_report(results)
            results['processing_stages']['reporting'] = report_results
            
            # Calculate total processing time
            end_time = time.time()
            processing_time = end_time - start_time
            results['processing_time'] = processing_time
            results['end_time'] = end_time
            results['success'] = True
            
            logger.info(f"\n✅ Pipeline completed successfully in {processing_time:.1f} seconds")
            logger.info(f"📁 Results saved to: {self.output_path}")
            
            return results
            
        except Exception as e:
            end_time = time.time()
            results['processing_time'] = end_time - start_time
            results['end_time'] = end_time
            results['error'] = str(e)
            logger.error(f"❌ Pipeline failed: {e}")
            return results
    
    def _run_core_detection(self) -> Dict:
        """Run core detection on both H&E and Orion images."""
        
        detection_results = {
            'success': False,
            'he_detection': None,
            'orion_detection': None,
            'error': None
        }
        
        try:
            # Detect cores in H&E image
            logger.info("Detecting cores in H&E image...")
            he_detection = self.detector.detect_cores(self.config.he_image_path, image_type="he")
            detection_results['he_detection'] = he_detection
            
            logger.info(f"H&E: Detected {he_detection['filtered_cores_count']} cores")
            
            # Detect cores in Orion image
            logger.info("Detecting cores in Orion image...")
            orion_detection = self.detector.detect_cores(self.config.orion_image_path, image_type="orion")
            detection_results['orion_detection'] = orion_detection
            
            logger.info(f"Orion: Detected {orion_detection['filtered_cores_count']} cores")
            
            # Check minimum core requirements
            total_cores = min(he_detection['filtered_cores_count'], orion_detection['filtered_cores_count'])
            if total_cores < self.config.min_cores_required:
                raise ValueError(f"Insufficient cores detected: {total_cores} < {self.config.min_cores_required} required")
            
            # Save detection results
            if self.config.save_intermediate_results:
                with open(self.detection_output_dir / "he_detection_results.json", 'w') as f:
                    # Create a serializable version
                    he_serializable = self._make_serializable(he_detection)
                    json.dump(he_serializable, f, indent=2)
                
                with open(self.detection_output_dir / "orion_detection_results.json", 'w') as f:
                    orion_serializable = self._make_serializable(orion_detection)
                    json.dump(orion_serializable, f, indent=2)
            
            # Create detection visualizations
            if self.config.create_visualizations:
                he_vis_path = self.visualization_output_dir / "he_core_detection.png"
                self.detector.visualize_detection(he_detection, str(he_vis_path))
                
                orion_vis_path = self.visualization_output_dir / "orion_core_detection.png"
                self.detector.visualize_detection(orion_detection, str(orion_vis_path))
            
            detection_results['success'] = True
            return detection_results
            
        except Exception as e:
            detection_results['error'] = str(e)
            logger.error(f"Core detection failed: {e}")
            return detection_results
    
    def _run_core_matching(self, detection_results: Dict) -> Dict:
        """Run core matching between H&E and Orion detections."""
        
        matching_results = {
            'success': False,
            'matching_data': None,
            'error': None
        }
        
        try:
            he_detection = detection_results['he_detection']
            orion_detection = detection_results['orion_detection']
            
            # Perform core matching
            logger.info("Matching cores between H&E and Orion images...")
            matching_data = self.matcher.match_cores(he_detection, orion_detection)
            matching_results['matching_data'] = matching_data
            
            # Log matching results
            matched_pairs = matching_data['high_quality_matches']
            he_match_rate = matching_data['matching_statistics']['match_rate_he']
            orion_match_rate = matching_data['matching_statistics']['match_rate_orion']
            
            logger.info(f"Successfully matched {matched_pairs} core pairs")
            logger.info(f"Match rates: H&E {he_match_rate:.1%}, Orion {orion_match_rate:.1%}")
            
            # Check minimum matches
            if matched_pairs < self.config.min_cores_required:
                raise ValueError(f"Insufficient matched cores: {matched_pairs} < {self.config.min_cores_required} required")
            
            # Save matching results
            if self.config.save_intermediate_results:
                with open(self.matching_output_dir / "core_matching_results.json", 'w') as f:
                    matching_serializable = self._make_serializable(matching_data)
                    json.dump(matching_serializable, f, indent=2)
            
            # Create matching visualization
            if self.config.create_visualizations:
                matching_vis_path = self.visualization_output_dir / "core_matching.png"
                self.matcher.visualize_matches(
                    matching_data, 
                    self.config.he_image_path, 
                    self.config.orion_image_path,
                    str(matching_vis_path)
                )
            
            matching_results['success'] = True
            return matching_results
            
        except Exception as e:
            matching_results['error'] = str(e)
            logger.error(f"Core matching failed: {e}")
            return matching_results
    
    def _run_core_extraction(self, detection_results: Dict, matching_results: Dict) -> Dict:
        """Run core extraction for matched pairs."""
        
        extraction_results = {
            'success': False,
            'he_extraction': None,
            'orion_extraction': None,
            'paired_cores': None,
            'error': None
        }
        
        try:
            he_detection = detection_results['he_detection']
            orion_detection = detection_results['orion_detection']
            matching_data = matching_results['matching_data']
            
            # Extract cores from both images
            logger.info("Extracting individual cores...")
            
            # Extract H&E cores
            he_extraction = self.extractor.extract_cores_from_detection(
                self.config.he_image_path, he_detection
            )
            extraction_results['he_extraction'] = he_extraction
            
            # Extract Orion cores (preserving all channels)
            orion_extraction = self.extractor.extract_cores_from_detection(
                self.config.orion_image_path, orion_detection
            )
            extraction_results['orion_extraction'] = orion_extraction
            
            # Create paired core information
            if self.config.create_paired_cores:
                paired_cores = self._create_paired_core_info(matching_data, he_extraction, orion_extraction)
                extraction_results['paired_cores'] = paired_cores
                
                # Save paired core information
                paired_cores_path = self.output_path / "paired_cores.csv"
                paired_df = pd.DataFrame(paired_cores)
                paired_df.to_csv(paired_cores_path, index=False)
                logger.info(f"Saved {len(paired_cores)} paired core records to {paired_cores_path}")
            
            logger.info(f"Extracted {len(he_extraction['cores_extracted'])} H&E cores")
            logger.info(f"Extracted {len(orion_extraction['cores_extracted'])} Orion cores")
            
            extraction_results['success'] = True
            return extraction_results
            
        except Exception as e:
            extraction_results['error'] = str(e)
            logger.error(f"Core extraction failed: {e}")
            return extraction_results
    
    def _create_paired_core_info(self, matching_data: Dict, he_extraction: Dict, orion_extraction: Dict) -> List[Dict]:
        """Create paired core information for matched cores."""
        
        # Create lookup dictionaries
        he_cores_by_id = {core['core_id']: core for core in he_extraction['cores_extracted']}
        orion_cores_by_id = {core['core_id']: core for core in orion_extraction['cores_extracted']}
        
        paired_cores = []
        
        for match in matching_data['matches']:
            he_core_id = f"core_{match['he_core_id']:03d}"
            orion_core_id = f"core_{match['orion_core_id']:03d}"
            
            he_core = he_cores_by_id.get(he_core_id)
            orion_core = orion_cores_by_id.get(orion_core_id)
            
            if he_core and orion_core:
                pair_info = {
                    'pair_id': f"pair_{len(paired_cores)+1:03d}",
                    'he_core_id': he_core_id,
                    'orion_core_id': orion_core_id,
                    'he_file_path': Path(he_core['file_path']).name,
                    'orion_file_path': Path(orion_core['file_path']).name,
                    'he_shape': str(he_core['extracted_shape']),
                    'orion_shape': str(orion_core['extracted_shape']),
                    'match_distance': match['distance'],
                    'match_confidence': match['confidence'],
                    'size_ratio': match['size_ratio'],
                    'he_quality_score': he_core.get('quality_score'),
                    'orion_quality_score': orion_core.get('quality_score')
                }
                paired_cores.append(pair_info)
        
        return paired_cores
    
    def _create_visualizations(self, detection_results: Dict, matching_results: Dict) -> Dict:
        """Create comprehensive visualizations."""
        
        visualization_results = {
            'success': False,
            'files_created': [],
            'error': None
        }
        
        try:
            files_created = []
            
            # Detection visualizations already created in detection stage
            he_detection_path = self.visualization_output_dir / "he_core_detection.png"
            orion_detection_path = self.visualization_output_dir / "orion_core_detection.png"
            matching_path = self.visualization_output_dir / "core_matching.png"
            
            if he_detection_path.exists():
                files_created.append(str(he_detection_path))
            if orion_detection_path.exists():
                files_created.append(str(orion_detection_path))
            if matching_path.exists():
                files_created.append(str(matching_path))
            
            # Create summary visualization
            summary_path = self.visualization_output_dir / "pipeline_summary.png"
            self._create_summary_visualization(detection_results, matching_results, str(summary_path))
            files_created.append(str(summary_path))
            
            visualization_results['files_created'] = files_created
            visualization_results['success'] = True
            
            logger.info(f"Created {len(files_created)} visualization files")
            return visualization_results
            
        except Exception as e:
            visualization_results['error'] = str(e)
            logger.error(f"Visualization creation failed: {e}")
            return visualization_results
    
    def _create_summary_visualization(self, detection_results: Dict, matching_results: Dict, output_path: str):
        """Create a summary visualization of the entire pipeline."""
        
        he_detection = detection_results['he_detection']
        orion_detection = detection_results['orion_detection']
        matching_data = matching_results['matching_data']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Core-First Pipeline Summary', fontsize=16, fontweight='bold')
        
        # Detection statistics
        detection_data = [he_detection['filtered_cores_count'], orion_detection['filtered_cores_count']]
        axes[0, 0].bar(['H&E', 'Orion'], detection_data, color=['red', 'blue'], alpha=0.7)
        axes[0, 0].set_title('Cores Detected per Image')
        axes[0, 0].set_ylabel('Number of Cores')
        for i, v in enumerate(detection_data):
            axes[0, 0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Matching statistics
        total_he = matching_data['total_he_cores']
        total_orion = matching_data['total_orion_cores']
        matched = matching_data['high_quality_matches']
        
        matching_data_plot = [matched, total_he - matched, total_orion - matched]
        labels = ['Matched', 'Unmatched H&E', 'Unmatched Orion']
        colors = ['green', 'red', 'blue']
        
        axes[0, 1].bar(labels, matching_data_plot, color=colors, alpha=0.7)
        axes[0, 1].set_title('Core Matching Results')
        axes[0, 1].set_ylabel('Number of Cores')
        axes[0, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(matching_data_plot):
            axes[0, 1].text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Quality metrics distribution (if available)
        if matching_data['matches']:
            confidences = [match['confidence'] for match in matching_data['matches']]
            axes[1, 0].hist(confidences, bins=10, alpha=0.7, color='green', edgecolor='black')
            axes[1, 0].set_title('Match Confidence Distribution')
            axes[1, 0].set_xlabel('Confidence Score')
            axes[1, 0].set_ylabel('Number of Matches')
            axes[1, 0].axvline(np.mean(confidences), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(confidences):.3f}')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'No matches found', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Match Confidence Distribution')
        
        # Pipeline timing (placeholder)
        stages = ['Detection', 'Matching', 'Extraction', 'Visualization']
        # These would be actual timings in a real implementation
        stage_times = [1.0, 0.5, 2.0, 0.3]  # Placeholder values
        
        axes[1, 1].pie(stage_times, labels=stages, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Processing Time Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_final_report(self, results: Dict) -> Dict:
        """Generate final pipeline report."""
        
        report_results = {
            'success': False,
            'report_path': None,
            'error': None
        }
        
        try:
            report_path = self.output_path / "pipeline_report.md"
            
            # Generate markdown report
            with open(report_path, 'w') as f:
                f.write("# Core-First TMA Processing Pipeline Report\n\n")
                
                # Basic info
                f.write("## Pipeline Configuration\n")
                f.write(f"- **H&E Image**: `{self.config.he_image_path}`\n")
                f.write(f"- **Orion Image**: `{self.config.orion_image_path}`\n")
                f.write(f"- **Output Directory**: `{self.config.output_dir}`\n")
                f.write(f"- **Processing Time**: {results.get('processing_time', 0):.1f} seconds\n\n")
                
                # Results summary
                if results['success']:
                    detection = results['processing_stages']['detection']
                    matching = results['processing_stages']['matching']
                    extraction = results['processing_stages']['extraction']
                    
                    f.write("## Results Summary\n")
                    f.write("### ✅ Core Detection\n")
                    f.write(f"- **H&E Cores Detected**: {detection['he_detection']['filtered_cores_count']}\n")
                    f.write(f"- **Orion Cores Detected**: {detection['orion_detection']['filtered_cores_count']}\n")
                    
                    f.write("\n### ✅ Core Matching\n")
                    matching_data = matching['matching_data']
                    f.write(f"- **Matched Pairs**: {matching_data['high_quality_matches']}\n")
                    f.write(f"- **H&E Match Rate**: {matching_data['matching_statistics']['match_rate_he']:.1%}\n")
                    f.write(f"- **Orion Match Rate**: {matching_data['matching_statistics']['match_rate_orion']:.1%}\n")
                    f.write(f"- **Mean Match Distance**: {matching_data['matching_statistics']['mean_distance']:.1f} pixels\n")
                    
                    f.write("\n### ✅ Core Extraction\n")
                    f.write(f"- **H&E Cores Extracted**: {len(extraction['he_extraction']['cores_extracted'])}\n")
                    f.write(f"- **Orion Cores Extracted**: {len(extraction['orion_extraction']['cores_extracted'])}\n")
                    
                    # Add channel information for Orion
                    orion_channel_info = extraction['orion_extraction']['channel_info']
                    if orion_channel_info:
                        f.write(f"- **Orion Channels Preserved**: {orion_channel_info['num_channels']}\n")
                    
                    f.write("\n## 🎯 Next Steps\n")
                    f.write("Your cores have been successfully extracted and are ready for:\n")
                    f.write("- Individual core registration (optional refinement)\n")
                    f.write("- Deep learning model training\n")
                    f.write("- Spatial analysis workflows\n")
                    f.write("- Integration with ROSIE baseline\n")
                
                else:
                    f.write("## ❌ Pipeline Failed\n")
                    f.write(f"**Error**: {results.get('error', 'Unknown error')}\n")
                
                f.write(f"\n---\n*Report generated by Core-First Pipeline*\n")
            
            # Also save as JSON for programmatic access
            json_report_path = self.output_path / "pipeline_report.json"
            with open(json_report_path, 'w') as f:
                json.dump(self._make_serializable(results), f, indent=2)
            
            report_results['report_path'] = str(report_path)
            report_results['success'] = True
            
            logger.info(f"Generated final report: {report_path}")
            return report_results
            
        except Exception as e:
            report_results['error'] = str(e)
            logger.error(f"Report generation failed: {e}")
            return report_results
    
    def _make_serializable(self, obj) -> Union[Dict, List, str, int, float, bool, None]:
        """Make objects JSON serializable by converting problematic types."""
        
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (Path, str)):
            return str(obj)
        else:
            return obj


def main():
    """Example usage of the CoreFirstPipeline."""
    
    # Configuration
    config = CoreFirstPipelineConfig(
        he_image_path="data/raw/TA118-HEraw.ome.tiff",
        orion_image_path="data/raw/TA118-Orionraw.ome.tiff",
        output_dir="core_first_output",
        create_visualizations=True,
        save_intermediate_results=True,
        create_paired_cores=True,
        min_cores_required=5
    )
    
    # Create and run pipeline
    pipeline = CoreFirstPipeline(config)
    results = pipeline.run()
    
    # Print summary
    if results['success']:
        print("\n🎉 Core-First Pipeline Completed Successfully!")
        
        # Extract key metrics
        detection = results['processing_stages']['detection']
        matching = results['processing_stages']['matching'] 
        extraction = results['processing_stages']['extraction']
        
        he_cores = detection['he_detection']['filtered_cores_count']
        orion_cores = detection['orion_detection']['filtered_cores_count'] 
        matched_pairs = matching['matching_data']['high_quality_matches']
        processing_time = results['processing_time']
        
        print(f"📊 Results:")
        print(f"   • H&E cores detected: {he_cores}")
        print(f"   • Orion cores detected: {orion_cores}")
        print(f"   • Matched core pairs: {matched_pairs}")
        print(f"   • Processing time: {processing_time:.1f} seconds")
        print(f"📁 Output directory: {config.output_dir}")
        
    else:
        print(f"\n❌ Pipeline Failed: {results.get('error', 'Unknown error')}")
        print("Check the logs for more details.")


if __name__ == "__main__":
    main() 