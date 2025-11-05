#!/usr/bin/env python3
"""
Test the improved SPACEc pipeline with better tissue extraction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spacec_codex_processor import SPACECCODEXProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the improved pipeline"""
    
    # Configuration
    data_root = "../../data/nomad_data/CODEX"
    output_root = "../../data/nomad_data/CODEX/processed_improved"
    
    logger.info("Testing improved SPACEc pipeline with adaptive tissue extraction")
    
    try:
        # Initialize processor
        processor = SPACECCODEXProcessor(data_root, output_root)
        
        # Test with just the first TMA to see if tissue extraction works
        if processor.tma_dirs:
            first_tma = processor.tma_dirs[0]
            logger.info(f"Testing with first TMA: {first_tma.name}")
            
            # Process single TMA
            result = processor.process_single_tma(first_tma)
            
            logger.info("Test completed!")
            logger.info(f"Result: {result}")
            
            if result.get("files_processed"):
                for file_result in result["files_processed"]:
                    logger.info(f"File: {file_result.get('file', 'Unknown')}")
                    logger.info(f"Status: {file_result.get('status', 'Unknown')}")
                    if file_result.get('n_tissues'):
                        logger.info(f"Tissues found: {file_result['n_tissues']}")
                    if file_result.get('total_cells'):
                        logger.info(f"Total cells: {file_result['total_cells']}")
        else:
            logger.error("No TMA directories found")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 