#!/usr/bin/env python3
"""
Optimize tissue extraction by systematically testing cutoff combinations
"""

import sys
import warnings
warnings.filterwarnings('ignore')

try:
    import spacec as sp
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import logging
    import pandas as pd
    from itertools import product
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def optimize_tissue_extraction(qptiff_file: Path, output_dir: Path):
    """
    Optimize tissue extraction by testing different cutoff combinations
    """
    logger.info(f"Optimizing tissue extraction for: {qptiff_file.name}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Downscale the image
        logger.info("Step 1: Downscaling image...")
        resized_im = sp.hf.downscale_tissue(
            file_path=str(qptiff_file),
            downscale_factor=64,
            padding=50,
            output_dir=str(output_dir)
        )
        
        logger.info(f"Downscaled image shape: {resized_im.shape}")
        logger.info(f"Downscaled image min/max: {resized_im.min():.3f}/{resized_im.max():.3f}")
        
        # Step 2: Create systematic cutoff combinations
        logger.info("Step 2: Creating systematic cutoff combinations...")
        
        # Create linspace for lower and upper cutoffs
        lower_cutoffs = np.linspace(0.01, 0.5, 20)  # 20 values from 0.01 to 0.5
        upper_cutoffs = np.linspace(0.5, 0.99, 20)  # 20 values from 0.5 to 0.99
        
        logger.info(f"Testing {len(lower_cutoffs)} x {len(upper_cutoffs)} = {len(lower_cutoffs) * len(upper_cutoffs)} combinations")
        
        # Step 3: Test all combinations
        results = []
        best_n_tissues = 0
        best_params = None
        best_tissueframe = None
        
        total_combinations = len(lower_cutoffs) * len(upper_cutoffs)
        current_combination = 0
        
        for lower_cutoff in lower_cutoffs:
            for upper_cutoff in upper_cutoffs:
                current_combination += 1
                
                # Skip invalid combinations (lower >= upper)
                if lower_cutoff >= upper_cutoff:
                    continue
                
                # Progress update
                if current_combination % 50 == 0:
                    logger.info(f"Progress: {current_combination}/{total_combinations} combinations tested")
                
                try:
                    tissueframe = sp.tl.label_tissue(
                        resized_im,
                        lower_cutoff=lower_cutoff,
                        upper_cutoff=upper_cutoff
                    )
                    
                    unique_regions = tissueframe['region'].unique()
                    n_tissues = len([r for r in unique_regions if r != 0])  # Exclude background
                    
                    results.append({
                        'lower_cutoff': lower_cutoff,
                        'upper_cutoff': upper_cutoff,
                        'n_tissues': n_tissues,
                        'success': True
                    })
                    
                    # Track the best result
                    if n_tissues > best_n_tissues:
                        best_n_tissues = n_tissues
                        best_params = (lower_cutoff, upper_cutoff)
                        best_tissueframe = tissueframe
                        logger.info(f"New best: {n_tissues} tissues with cutoffs ({lower_cutoff:.3f}, {upper_cutoff:.3f})")
                    
                except Exception as e:
                    results.append({
                        'lower_cutoff': lower_cutoff,
                        'upper_cutoff': upper_cutoff,
                        'n_tissues': 0,
                        'success': False,
                        'error': str(e)
                    })
        
        # Step 4: Analyze results
        logger.info("Step 4: Analyzing results...")
        
        results_df = pd.DataFrame(results)
        successful_results = results_df[results_df['success'] == True]
        
        logger.info(f"Total combinations tested: {len(results_df)}")
        logger.info(f"Successful combinations: {len(successful_results)}")
        logger.info(f"Combinations with tissues: {len(successful_results[successful_results['n_tissues'] > 0])}")
        
        if len(successful_results) > 0:
            logger.info(f"Best result: {best_n_tissues} tissues with cutoffs {best_params}")
            
            # Find top 10 results
            top_results = successful_results.nlargest(10, 'n_tissues')
            logger.info("\nTop 10 results:")
            for i, row in top_results.iterrows():
                logger.info(f"  {row['n_tissues']} tissues: ({row['lower_cutoff']:.3f}, {row['upper_cutoff']:.3f})")
            
            # Save results
            results_file = output_dir / "tissue_extraction_results.csv"
            results_df.to_csv(results_file, index=False)
            logger.info(f"Results saved to: {results_file}")
            
            # Create visualization
            try:
                create_visualization(results_df, output_dir)
            except Exception as e:
                logger.warning(f"Could not create visualization: {e}")
            
            return True, best_tissueframe, best_params, results_df
            
        else:
            logger.error("No successful tissue extraction found")
            return False, None, None, results_df
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None

def create_visualization(results_df, output_dir):
    """
    Create visualization of the results
    """
    successful_results = results_df[results_df['success'] == True]
    
    if len(successful_results) == 0:
        return
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Create pivot table for heatmap
    pivot_data = successful_results.pivot_table(
        values='n_tissues', 
        index='lower_cutoff', 
        columns='upper_cutoff', 
        fill_value=0
    )
    
    # Plot heatmap
    plt.imshow(pivot_data.values, cmap='viridis', aspect='auto', 
               extent=[pivot_data.columns.min(), pivot_data.columns.max(),
                      pivot_data.index.min(), pivot_data.index.max()])
    
    plt.colorbar(label='Number of Tissue Pieces')
    plt.xlabel('Upper Cutoff')
    plt.ylabel('Lower Cutoff')
    plt.title('Tissue Extraction Results: Number of Tissue Pieces vs Cutoffs')
    
    # Add best result marker
    best_result = successful_results.loc[successful_results['n_tissues'].idxmax()]
    plt.scatter(best_result['upper_cutoff'], best_result['lower_cutoff'], 
               color='red', s=100, marker='*', label=f'Best: {best_result["n_tissues"]} tissues')
    plt.legend()
    
    # Save plot
    plot_file = output_dir / "tissue_extraction_heatmap.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to: {plot_file}")

def test_optimal_parameters(qptiff_file: Path, output_dir: Path, best_params):
    """
    Test the optimal parameters by extracting tissues
    """
    if best_params is None:
        logger.error("No optimal parameters to test")
        return False
    
    lower_cutoff, upper_cutoff = best_params
    logger.info(f"Testing optimal parameters: ({lower_cutoff:.3f}, {upper_cutoff:.3f})")
    
    try:
        # Downscale image
        resized_im = sp.hf.downscale_tissue(
            file_path=str(qptiff_file),
            downscale_factor=64,
            padding=50,
            output_dir=str(output_dir)
        )
        
        # Extract tissues with optimal parameters
        tissueframe = sp.tl.label_tissue(
            resized_im,
            lower_cutoff=lower_cutoff,
            upper_cutoff=upper_cutoff
        )
        
        # Extract individual tissues
        extracted_tissues = []
        unique_regions = tissueframe['region'].unique()
        
        for region in unique_regions:
            if region != 0:  # Skip background
                try:
                    sp.tl.save_labelled_tissue(
                        filepath=str(qptiff_file),
                        tissueframe=tissueframe,
                        output_dir=str(output_dir),
                        downscale_factor=64,
                        region=region,
                        padding=50
                    )
                    
                    # Find the extracted tissue file
                    tissue_files = list(output_dir.glob(f"reg{region:03d}_*.tif"))
                    if tissue_files:
                        extracted_tissues.extend(tissue_files)
                        
                except Exception as e:
                    logger.error(f"Error extracting tissue region {region}: {e}")
                    continue
        
        logger.info(f"Successfully extracted {len(extracted_tissues)} tissue pieces")
        return len(extracted_tissues) > 0
        
    except Exception as e:
        logger.error(f"Error testing optimal parameters: {e}")
        return False

def main():
    """Main function"""
    
    # Find a qptiff file to test
    data_root = Path("../../data/nomad_data/CODEX")
    
    # Find TMA directories
    tma_dirs = [d for d in data_root.iterdir() if d.is_dir() and "TMA" in d.name]
    
    if not tma_dirs:
        logger.error("No TMA directories found")
        return
    
    # Use the first TMA
    tma_dir = tma_dirs[0]
    logger.info(f"Using TMA: {tma_dir.name}")
    
    # Find qptiff files
    scan_dir = tma_dir / "Scan1"
    if not scan_dir.exists():
        logger.error(f"Scan1 directory not found in {tma_dir}")
        return
    
    qptiff_files = list(scan_dir.glob("*.qptiff"))
    if not qptiff_files:
        logger.error("No qptiff files found")
        return
    
    # Use the first qptiff file
    qptiff_file = qptiff_files[0]
    logger.info(f"Using file: {qptiff_file.name}")
    
    # Create output directory
    output_dir = Path("../../data/nomad_data/CODEX/optimization_output")
    
    # Run optimization
    success, tissueframe, best_params, results_df = optimize_tissue_extraction(qptiff_file, output_dir)
    
    if success and best_params:
        logger.info(f"Optimization successful!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Number of tissues: {len([r for r in tissueframe['region'].unique() if r != 0])}")
        
        # Test the optimal parameters
        test_success = test_optimal_parameters(qptiff_file, output_dir, best_params)
        
        if test_success:
            logger.info("Optimal parameters work! Tissue extraction successful.")
        else:
            logger.warning("Optimal parameters found but tissue extraction failed.")
    else:
        logger.error("Optimization failed - no valid parameters found")

if __name__ == "__main__":
    main() 