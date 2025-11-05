"""
Matched Core Registration Pipeline

Integrates dual-modality core detection with VALIS registration for robust
TMA core pairing and alignment.

This module extends the existing registration pipeline to:
1. Detect cores in both H&E and Orion images using SpaceC
2. Match corresponding cores using spatial optimization
3. Extract matched core pairs 
4. Register each pair individually with VALIS
5. Generate training datasets with quality control
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tifffile import imread, imwrite
import json
from openslide import OpenSlide

# Import existing registration infrastructure
sys.path.append('..')
try:
    from registration_pipeline import (
        RegistrationConfig, VALISRegistrar, 
        DatasetPreparator, QualityController
    )
    REGISTRATION_AVAILABLE = True
except ImportError:
    REGISTRATION_AVAILABLE = False
    print("Warning: registration_pipeline.py not found. Using simplified version.")

from .dual_modality_core_detector import DualModalityCoreDetector, CoreDetectionConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchedCoreRegistrationPipeline:
    """
    Complete TMA core pairing and registration pipeline.
    
    Combines dual-modality core detection with individual core registration
    using VALIS for maximum accuracy.
    """
    
    def __init__(self, 
                 detection_config: CoreDetectionConfig,
                 registration_config: Optional[RegistrationConfig] = None):
        """
        Initialize the matched core registration pipeline.
        
        Args:
            detection_config: Configuration for core detection
            registration_config: Configuration for VALIS registration
        """
        self.detection_config = detection_config
        
        # Initialize core detector
        self.core_detector = DualModalityCoreDetector(detection_config)
        
        # Initialize registration components if available
        if REGISTRATION_AVAILABLE and registration_config:
            self.registration_config = registration_config
            self.registrar = VALISRegistrar(registration_config)
            self.preparator = DatasetPreparator(registration_config)
            self.qc = QualityController(registration_config)
        else:
            self.registration_config = None
            logger.warning("Registration pipeline not available. Core detection only.")
        
        # Create output directories
        self.output_path = Path(detection_config.temp_dir or "./tma_pipeline_output")
        self.output_path.mkdir(exist_ok=True)
        
        (self.output_path / "core_detection").mkdir(exist_ok=True)
        (self.output_path / "extracted_cores").mkdir(exist_ok=True)
        (self.output_path / "registered_cores").mkdir(exist_ok=True)
        (self.output_path / "training_pairs").mkdir(exist_ok=True)
        (self.output_path / "quality_reports").mkdir(exist_ok=True)
    
    def run_complete_pipeline(self, he_wsi_path: str, orion_wsi_path: str) -> Dict:
        """
        Run the complete TMA core pairing and registration pipeline.
        
        Args:
            he_wsi_path: Path to H&E whole slide image
            orion_wsi_path: Path to Orion whole slide image
            
        Returns:
            Complete pipeline results dictionary
        """
        logger.info("="*80)
        logger.info("STARTING TMA CORE PAIRING AND REGISTRATION PIPELINE")
        logger.info("="*80)
        
        pipeline_results = {
            'input_files': {
                'he_wsi_path': he_wsi_path,
                'orion_wsi_path': orion_wsi_path
            },
            'pipeline_stages': {},
            'final_statistics': {}
        }
        
        try:
            # Stage 1: Core Detection and Matching
            logger.info("\n" + "="*50)
            logger.info("STAGE 1: DUAL-MODALITY CORE DETECTION")
            logger.info("="*50)
            
            detection_results = self.core_detector.detect_and_match_cores(
                he_wsi_path, orion_wsi_path
            )
            pipeline_results['pipeline_stages']['detection'] = detection_results
            
            if not detection_results['success']:
                raise RuntimeError(f"Core detection failed: {detection_results.get('error', 'Unknown error')}")
            
            matched_cores = detection_results['matched_cores']
            logger.info(f"Successfully matched {len(matched_cores)} core pairs")
            
            # Stage 2: Core Extraction
            logger.info("\n" + "="*50)
            logger.info("STAGE 2: CORE EXTRACTION")
            logger.info("="*50)
            
            extraction_results = self._extract_matched_core_pairs(
                he_wsi_path, orion_wsi_path, matched_cores
            )
            pipeline_results['pipeline_stages']['extraction'] = extraction_results
            
            # Stage 3: Individual Core Registration (if VALIS available)
            if self.registration_config:
                logger.info("\n" + "="*50)
                logger.info("STAGE 3: INDIVIDUAL CORE REGISTRATION")
                logger.info("="*50)
                
                registration_results = self._register_core_pairs(
                    extraction_results['core_pairs']
                )
                pipeline_results['pipeline_stages']['registration'] = registration_results
                
                # Stage 4: Training Dataset Preparation
                logger.info("\n" + "="*50)
                logger.info("STAGE 4: TRAINING DATASET PREPARATION")
                logger.info("="*50)
                
                dataset_results = self._prepare_training_dataset(registration_results)
                pipeline_results['pipeline_stages']['dataset_preparation'] = dataset_results
            
            # Stage 5: Generate Final Report
            logger.info("\n" + "="*50)
            logger.info("STAGE 5: FINAL QUALITY REPORT")
            logger.info("="*50)
            
            final_report = self._generate_final_report(pipeline_results)
            pipeline_results['final_report'] = final_report
            
            # Save complete results
            self._save_pipeline_results(pipeline_results)
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_results['success'] = False
            pipeline_results['error'] = str(e)
        
        return pipeline_results
    
    def _extract_matched_core_pairs(self, he_wsi_path: str, orion_wsi_path: str,
                                   matched_cores: List[Dict]) -> Dict:
        """
        Extract matched core pairs from whole slide images.
        
        Args:
            he_wsi_path: Path to H&E WSI
            orion_wsi_path: Path to Orion WSI  
            matched_cores: List of matched core information
            
        Returns:
            Extraction results with core pair file paths
        """
        logger.info(f"Extracting {len(matched_cores)} matched core pairs...")
        
        extraction_results = {
            'core_pairs': [],
            'extraction_stats': {
                'total_matched': len(matched_cores),
                'successfully_extracted': 0,
                'failed_extractions': 0
            }
        }
        
        # Open whole slide images
        he_slide = OpenSlide(he_wsi_path)
        orion_img = imread(orion_wsi_path)
        
        core_pairs_dir = self.output_path / "extracted_cores"
        
        for i, match_info in enumerate(matched_cores):
            try:
                # Generate core identifiers
                he_core_id = f"reg{match_info['he_core_id']:03d}"
                orion_core_id = f"reg{match_info['orion_core_id']:03d}"
                pair_id = f"pair_{i:03d}_{he_core_id}_{orion_core_id}"
                
                # Extract H&E core
                he_bbox = match_info['he_bbox']  # (x_min, y_min, x_max, y_max)
                he_width = he_bbox[2] - he_bbox[0]
                he_height = he_bbox[3] - he_bbox[1]
                
                he_core = he_slide.read_region(
                    (he_bbox[0], he_bbox[1]), 0, (he_width, he_height)
                ).convert("RGB")
                
                he_core_path = core_pairs_dir / f"{pair_id}_HE.tif"
                he_core.save(he_core_path)
                
                # Extract Orion core
                orion_bbox = match_info['orion_bbox']
                orion_core = orion_img[
                    :, 
                    orion_bbox[1]:orion_bbox[3],  # y_min:y_max
                    orion_bbox[0]:orion_bbox[2]   # x_min:x_max
                ]
                
                orion_core_path = core_pairs_dir / f"{pair_id}_Orion.tif"
                imwrite(orion_core_path, orion_core)
                
                # Store core pair information
                extraction_results['core_pairs'].append({
                    'pair_id': pair_id,
                    'he_core_path': str(he_core_path),
                    'orion_core_path': str(orion_core_path),
                    'he_core_id': match_info['he_core_id'],
                    'orion_core_id': match_info['orion_core_id'],
                    'match_info': match_info
                })
                
                extraction_results['extraction_stats']['successfully_extracted'] += 1
                
            except Exception as e:
                logger.error(f"Failed to extract core pair {i}: {e}")
                extraction_results['extraction_stats']['failed_extractions'] += 1
                continue
        
        he_slide.close()
        
        logger.info(f"Successfully extracted {extraction_results['extraction_stats']['successfully_extracted']} core pairs")
        return extraction_results
    
    def _register_core_pairs(self, core_pairs: List[Dict]) -> Dict:
        """
        Register individual core pairs using VALIS.
        
        Args:
            core_pairs: List of extracted core pair information
            
        Returns:
            Registration results for all core pairs
        """
        logger.info(f"Registering {len(core_pairs)} core pairs with VALIS...")
        
        registration_results = {
            'registered_pairs': [],
            'registration_stats': {
                'total_pairs': len(core_pairs),
                'successful_registrations': 0,
                'failed_registrations': 0
            }
        }
        
        # Determine if we should use parallel processing
        use_parallel = len(core_pairs) > 5 and self.registration_config.num_workers > 1
        
        if use_parallel:
            # Parallel registration
            with ProcessPoolExecutor(max_workers=self.registration_config.num_workers) as executor:
                futures = []
                
                for core_pair in core_pairs:
                    future = executor.submit(
                        self._register_single_core_pair, 
                        core_pair
                    )
                    futures.append((future, core_pair['pair_id']))
                
                # Collect results
                for future, pair_id in as_completed(futures):
                    try:
                        result = future.result(timeout=600)  # 10 minute timeout
                        registration_results['registered_pairs'].append(result)
                        
                        if result['success']:
                            registration_results['registration_stats']['successful_registrations'] += 1
                        else:
                            registration_results['registration_stats']['failed_registrations'] += 1
                            
                    except Exception as e:
                        logger.error(f"Registration failed for {pair_id}: {e}")
                        registration_results['registration_stats']['failed_registrations'] += 1
        else:
            # Sequential registration
            for core_pair in core_pairs:
                try:
                    result = self._register_single_core_pair(core_pair)
                    registration_results['registered_pairs'].append(result)
                    
                    if result['success']:
                        registration_results['registration_stats']['successful_registrations'] += 1
                    else:
                        registration_results['registration_stats']['failed_registrations'] += 1
                        
                except Exception as e:
                    logger.error(f"Registration failed for {core_pair['pair_id']}: {e}")
                    registration_results['registration_stats']['failed_registrations'] += 1
        
        success_rate = (registration_results['registration_stats']['successful_registrations'] / 
                       len(core_pairs) * 100 if len(core_pairs) > 0 else 0)
        
        logger.info(f"Registration completed: {registration_results['registration_stats']['successful_registrations']}/{len(core_pairs)} pairs ({success_rate:.1f}% success rate)")
        
        return registration_results
    
    def _register_single_core_pair(self, core_pair: Dict) -> Dict:
        """Register a single core pair using VALIS."""
        
        try:
            result = self.registrar.register_core_pair(
                core_pair['he_core_path'],
                core_pair['orion_core_path'],
                core_pair['pair_id']
            )
            
            # Add core pair metadata
            result['core_pair_info'] = core_pair
            
            # Save registered result if successful
            if result['success']:
                registered_dir = self.output_path / "registered_cores"
                
                # Save registered H&E (reference doesn't change)
                he_registered_path = registered_dir / f"{core_pair['pair_id']}_HE_registered.tif"
                imwrite(he_registered_path, result['he_img'])
                result['he_registered_path'] = str(he_registered_path)
                
                # Save warped Orion
                orion_warped_path = registered_dir / f"{core_pair['pair_id']}_Orion_warped.tif"
                imwrite(orion_warped_path, result['warped_orion'])
                result['orion_warped_path'] = str(orion_warped_path)
            
            return result
            
        except Exception as e:
            return {
                'pair_id': core_pair['pair_id'],
                'success': False,
                'error': str(e),
                'core_pair_info': core_pair
            }
    
    def _prepare_training_dataset(self, registration_results: Dict) -> Dict:
        """
        Prepare training dataset from successfully registered core pairs.
        
        Args:
            registration_results: Results from core pair registration
            
        Returns:
            Training dataset preparation results
        """
        logger.info("Preparing training dataset from registered cores...")
        
        successful_registrations = [
            result for result in registration_results['registered_pairs'] 
            if result['success']
        ]
        
        if len(successful_registrations) == 0:
            logger.warning("No successful registrations available for training dataset")
            return {'success': False, 'error': 'No successful registrations'}
        
        # Use existing dataset preparator
        pairs_dir = self.preparator.create_training_pairs(successful_registrations)
        
        # Generate quality report
        quality_df = self.qc.assess_registration_quality(registration_results['registered_pairs'])
        quality_report_path = self.output_path / "quality_reports" / "registration_quality.csv"
        quality_df.to_csv(quality_report_path, index=False)
        
        # Create quality plots
        self.qc.create_quality_plots(
            registration_results['registered_pairs'], 
            str(self.output_path / "quality_reports")
        )
        
        return {
            'success': True,
            'training_pairs_directory': pairs_dir,
            'quality_report_path': str(quality_report_path),
            'num_training_pairs': len(successful_registrations)
        }
    
    def _generate_final_report(self, pipeline_results: Dict) -> Dict:
        """Generate comprehensive final report."""
        
        detection_stats = pipeline_results['pipeline_stages']['detection']['detection_stats']
        extraction_stats = pipeline_results['pipeline_stages']['extraction']['extraction_stats']
        
        final_report = {
            'pipeline_summary': {
                'input_files': pipeline_results['input_files'],
                'total_stages_completed': len(pipeline_results['pipeline_stages']),
                'overall_success': True
            },
            'core_detection_summary': {
                'he_cores_detected': detection_stats['he_cores_detected'],
                'orion_cores_detected': detection_stats['orion_cores_detected'],
                'matched_cores': detection_stats['matched_cores'],
                'matching_rate': detection_stats['matched_cores'] / min(
                    detection_stats['he_cores_detected'], 
                    detection_stats['orion_cores_detected']
                ) if min(detection_stats['he_cores_detected'], detection_stats['orion_cores_detected']) > 0 else 0
            },
            'extraction_summary': {
                'cores_successfully_extracted': extraction_stats['successfully_extracted'],
                'extraction_success_rate': extraction_stats['successfully_extracted'] / extraction_stats['total_matched'] if extraction_stats['total_matched'] > 0 else 0
            }
        }
        
        # Add registration summary if available
        if 'registration' in pipeline_results['pipeline_stages']:
            reg_stats = pipeline_results['pipeline_stages']['registration']['registration_stats']
            final_report['registration_summary'] = {
                'cores_successfully_registered': reg_stats['successful_registrations'],
                'registration_success_rate': reg_stats['successful_registrations'] / reg_stats['total_pairs'] if reg_stats['total_pairs'] > 0 else 0,
                'total_registration_pairs': reg_stats['total_pairs']
            }
        
        # Add training dataset summary if available
        if 'dataset_preparation' in pipeline_results['pipeline_stages']:
            dataset_info = pipeline_results['pipeline_stages']['dataset_preparation']
            if dataset_info['success']:
                final_report['training_dataset_summary'] = {
                    'training_pairs_created': dataset_info['num_training_pairs'],
                    'training_pairs_directory': dataset_info['training_pairs_directory']
                }
        
        return final_report
    
    def _save_pipeline_results(self, pipeline_results: Dict):
        """Save complete pipeline results to disk."""
        
        # Save main results as JSON
        results_path = self.output_path / "pipeline_results.json"
        
        # Make results JSON serializable
        json_results = self._make_json_serializable(pipeline_results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Pipeline results saved to: {results_path}")
        
        # Save summary report
        if 'final_report' in pipeline_results:
            summary_path = self.output_path / "pipeline_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(pipeline_results['final_report'], f, indent=2)
            
            logger.info(f"Pipeline summary saved to: {summary_path}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        self.core_detector.cleanup()
        logger.info("Pipeline cleanup completed")


def main():
    """Example usage of the matched core registration pipeline."""
    
    # Example configuration
    detection_config = CoreDetectionConfig(
        downscale_factor=64,
        padding=50,
        he_lower_cutoff=0.15,
        he_upper_cutoff=0.25,
        orion_lower_cutoff=0.10,
        orion_upper_cutoff=0.20,
        save_debug_images=True
    )
    
    # Registration configuration (if registration_pipeline.py is available)
    if REGISTRATION_AVAILABLE:
        registration_config = RegistrationConfig(
            input_dir="./extracted_cores",
            output_dir="./registration_output",
            patch_size=256,
            stride=256,
            num_workers=4
        )
    else:
        registration_config = None
    
    # Create and run pipeline
    pipeline = MatchedCoreRegistrationPipeline(detection_config, registration_config)
    
    # Example file paths - replace with your actual paths
    he_wsi_path = "/path/to/your/he_image.ome.tiff"
    orion_wsi_path = "/path/to/your/orion_image.ome.tiff"
    
    try:
        results = pipeline.run_complete_pipeline(he_wsi_path, orion_wsi_path)
        
        print("\n" + "="*80)
        print("PIPELINE RESULTS SUMMARY")
        print("="*80)
        
        if 'final_report' in results:
            report = results['final_report']
            print(f"H&E cores detected: {report['core_detection_summary']['he_cores_detected']}")
            print(f"Orion cores detected: {report['core_detection_summary']['orion_cores_detected']}")
            print(f"Matched cores: {report['core_detection_summary']['matched_cores']}")
            print(f"Matching rate: {report['core_detection_summary']['matching_rate']:.1%}")
            
            if 'registration_summary' in report:
                print(f"Successfully registered: {report['registration_summary']['cores_successfully_registered']}")
                print(f"Registration success rate: {report['registration_summary']['registration_success_rate']:.1%}")
            
            if 'training_dataset_summary' in report:
                print(f"Training pairs created: {report['training_dataset_summary']['training_pairs_created']}")
                print(f"Training directory: {report['training_dataset_summary']['training_pairs_directory']}")
        
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main() 