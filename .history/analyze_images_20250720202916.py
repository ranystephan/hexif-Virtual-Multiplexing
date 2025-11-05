#!/usr/bin/env python3
"""
Image Analysis Tool for Registration Pipeline

This script analyzes H&E and Orion image pairs to identify potential issues
that could cause VALIS registration failures.
"""

import os
import pathlib
import pandas as pd
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_single_image(image_path: str) -> Dict:
    """Analyze a single image for potential issues."""
    try:
        img = imread(image_path)
        
        if img is None:
            return {'error': 'Failed to load image', 'valid': False}
        
        # Basic properties
        analysis = {
            'path': str(image_path),
            'shape': img.shape,
            'dtype': str(img.dtype),
            'size_mb': img.nbytes / (1024 * 1024),
            'valid': True,
            'issues': []
        }
        
        # Check for problematic dimensions
        if any(dim <= 0 for dim in img.shape):
            analysis['issues'].append(f"Invalid dimensions: {img.shape}")
            analysis['valid'] = False
        
        # Check minimum size
        min_size = 100
        if img.shape[0] < min_size or img.shape[1] < min_size:
            analysis['issues'].append(f"Too small: {img.shape}, minimum: {min_size}x{min_size}")
            analysis['valid'] = False
        
        # Check aspect ratio
        height, width = img.shape[:2]
        aspect_ratio = max(height/width, width/height)
        analysis['aspect_ratio'] = aspect_ratio
        
        if aspect_ratio > 50:
            analysis['issues'].append(f"Extreme aspect ratio: {aspect_ratio:.1f}")
        
        # Analyze intensity
        if img.ndim == 3 and img.shape[0] <= 10:  # Multi-channel, channel-first
            # Use first channel for analysis
            img_2d = img[0]
        elif img.ndim == 3:  # RGB or channel-last
            img_2d = np.mean(img, axis=2)
        else:
            img_2d = img
        
        analysis['mean_intensity'] = float(np.mean(img_2d))
        analysis['std_intensity'] = float(np.std(img_2d))
        analysis['min_intensity'] = float(np.min(img_2d))
        analysis['max_intensity'] = float(np.max(img_2d))
        
        # Check for problematic intensity distributions
        if analysis['mean_intensity'] < 5:
            analysis['issues'].append("Very dark image (mean < 5)")
        elif analysis['mean_intensity'] > 250:
            analysis['issues'].append("Very bright image (mean > 250)")
        
        if analysis['std_intensity'] < 1:
            analysis['issues'].append("Very low contrast (std < 1)")
        
        # Check for constant images
        if analysis['min_intensity'] == analysis['max_intensity']:
            analysis['issues'].append("Constant intensity image")
            analysis['valid'] = False
        
        return analysis
        
    except Exception as e:
        return {
            'path': str(image_path),
            'error': str(e),
            'valid': False,
            'issues': [f"Loading error: {e}"]
        }


def analyze_image_pairs(input_dir: str, he_suffix: str = "_HE.tif", orion_suffix: str = "_Orion.tif") -> pd.DataFrame:
    """Analyze all image pairs in a directory."""
    input_path = pathlib.Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find H&E images
    he_files = list(input_path.glob(f"*{he_suffix}"))
    logger.info(f"Found {len(he_files)} H&E images")
    
    results = []
    
    for he_file in he_files:
        # Extract core ID
        filename = he_file.stem
        if filename.endswith("_HE"):
            core_id = filename[:-3]
        else:
            core_id = filename.replace(he_suffix.replace(".tif", ""), "")
        
        # Find corresponding Orion file
        orion_file = input_path / f"{core_id}{orion_suffix}"
        
        if not orion_file.exists():
            results.append({
                'core_id': core_id,
                'he_valid': False,
                'orion_valid': False,
                'pair_valid': False,
                'issues': ['Missing Orion file'],
                'he_path': str(he_file),
                'orion_path': str(orion_file)
            })
            continue
        
        # Analyze both images
        he_analysis = analyze_single_image(he_file)
        orion_analysis = analyze_single_image(orion_file)
        
        # Combine results
        pair_result = {
            'core_id': core_id,
            'he_path': str(he_file),
            'orion_path': str(orion_file),
            'he_valid': he_analysis['valid'],
            'orion_valid': orion_analysis['valid'],
            'pair_valid': he_analysis['valid'] and orion_analysis['valid'],
            'issues': []
        }
        
        # Add H&E specific info
        if 'shape' in he_analysis:
            pair_result['he_shape'] = str(he_analysis['shape'])
            pair_result['he_mean_intensity'] = he_analysis['mean_intensity']
            pair_result['he_aspect_ratio'] = he_analysis.get('aspect_ratio', 1.0)
        
        # Add Orion specific info
        if 'shape' in orion_analysis:
            pair_result['orion_shape'] = str(orion_analysis['shape'])
            pair_result['orion_mean_intensity'] = orion_analysis['mean_intensity']
            pair_result['orion_aspect_ratio'] = orion_analysis.get('aspect_ratio', 1.0)
        
        # Combine issues
        if he_analysis.get('issues'):
            pair_result['issues'].extend([f"H&E: {issue}" for issue in he_analysis['issues']])
        
        if orion_analysis.get('issues'):
            pair_result['issues'].extend([f"Orion: {issue}" for issue in orion_analysis['issues']])
        
        if 'error' in he_analysis:
            pair_result['issues'].append(f"H&E error: {he_analysis['error']}")
        
        if 'error' in orion_analysis:
            pair_result['issues'].append(f"Orion error: {orion_analysis['error']}")
        
        results.append(pair_result)
    
    return pd.DataFrame(results)


def create_analysis_report(df: pd.DataFrame, output_dir: str = "./image_analysis"):
    """Create detailed analysis report."""
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save full results
    df.to_csv(output_path / "image_analysis_full.csv", index=False)
    
    # Create summary
    summary = {
        'total_pairs': len(df),
        'valid_pairs': df['pair_valid'].sum(),
        'invalid_pairs': (~df['pair_valid']).sum(),
        'he_issues': (~df['he_valid']).sum(),
        'orion_issues': (~df['orion_valid']).sum()
    }
    
    print("\n" + "="*60)
    print("IMAGE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total image pairs: {summary['total_pairs']}")
    print(f"Valid pairs: {summary['valid_pairs']}")
    print(f"Invalid pairs: {summary['invalid_pairs']}")
    print(f"H&E issues: {summary['he_issues']}")
    print(f"Orion issues: {summary['orion_issues']}")
    
    if summary['invalid_pairs'] > 0:
        print(f"\nSuccess rate: {summary['valid_pairs']/summary['total_pairs']:.1%}")
    
    # Show problematic images
    invalid_df = df[~df['pair_valid']]
    if len(invalid_df) > 0:
        print(f"\nPROBLEMATIC IMAGES:")
        print("-" * 40)
        
        for _, row in invalid_df.head(10).iterrows():
            print(f"Core {row['core_id']}:")
            if row['issues']:
                for issue in row['issues']:
                    print(f"  - {issue}")
            print()
        
        if len(invalid_df) > 10:
            print(f"... and {len(invalid_df) - 10} more problematic pairs")
    
    # Save problematic images list
    if len(invalid_df) > 0:
        invalid_df.to_csv(output_path / "problematic_images.csv", index=False)
        print(f"\nDetailed problematic images saved to: {output_path / 'problematic_images.csv'}")
    
    # Create recommendations
    recommendations = []
    
    if summary['invalid_pairs'] > summary['total_pairs'] * 0.5:
        recommendations.append("High failure rate detected. Check image format and quality.")
    
    # Check for common issues
    all_issues = []
    for issues_list in df['issues']:
        if isinstance(issues_list, list):
            all_issues.extend(issues_list)
    
    issue_counts = {}
    for issue in all_issues:
        issue_type = issue.split(':')[0] if ':' in issue else issue
        issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
    
    if issue_counts:
        print(f"\nMOST COMMON ISSUES:")
        print("-" * 40)
        for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{issue}: {count} images")
    
    # Save summary
    import json
    with open(output_path / "analysis_summary.json", 'w') as f:
        # Convert numpy types for JSON serialization
        summary_json = {}
        for k, v in summary.items():
            if hasattr(v, 'item'):  # numpy scalar
                summary_json[k] = v.item()
            else:
                summary_json[k] = v
        json.dump(summary_json, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to: {output_path}")
    return summary


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze images for registration pipeline compatibility")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing image pairs")
    parser.add_argument("--output_dir", type=str, default="./image_analysis", help="Output directory for analysis results")
    parser.add_argument("--he_suffix", type=str, default="_HE.tif", help="Suffix for H&E images")
    parser.add_argument("--orion_suffix", type=str, default="_Orion.tif", help="Suffix for Orion images")
    
    args = parser.parse_args()
    
    try:
        # Analyze images
        df = analyze_image_pairs(args.input_dir, args.he_suffix, args.orion_suffix)
        
        # Create report
        summary = create_analysis_report(df, args.output_dir)
        
        # Provide recommendations
        print(f"\nRECOMMENDATIONS:")
        print("-" * 40)
        
        valid_rate = summary['valid_pairs'] / summary['total_pairs']
        
        if valid_rate > 0.8:
            print("✓ Most images look good for registration")
        elif valid_rate > 0.5:
            print("⚠ Some images may cause registration issues")
            print("  Consider filtering out problematic images")
        else:
            print("✗ Many images have issues that will likely cause registration failures")
            print("  Review image format, quality, and preprocessing")
        
        print(f"\nTo run registration with only valid images, use:")
        print(f"  python run_registration.py --input_dir {args.input_dir} --output_dir ./registration_output")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 