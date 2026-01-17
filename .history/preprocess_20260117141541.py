#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from hexif.preprocessing import register_slides, detect_cores, extract_patches

def main():
    parser = argparse.ArgumentParser(description="Preprocess slides for HEXIF training or inference.")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract tissue cores from H&E (and optional Orion)")
    extract_parser.add_argument("--he_slide", required=True, help="Path to H&E slide")
    extract_parser.add_argument("--orion_slide", help="Path to Orion slide (optional)")
    extract_parser.add_argument("--output_dir", required=True, help="Output directory for NPY files")
    extract_parser.add_argument("--target_size", type=int, default=2048, help="Size of extracted cores (default 2048)")
    
    # Register command
    reg_parser = subparsers.add_parser("register", help="Register H&E and Orion slides")
    reg_parser.add_argument("--he_slide", required=True, help="Path to H&E slide")
    reg_parser.add_argument("--orion_slide", required=True, help="Path to Orion slide")
    reg_parser.add_argument("--output_dir", required=True, help="Output directory for registered slides")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    if args.command == "extract":
        logging.info("Detecting cores...")
        bboxes, _ = detect_cores(args.he_slide)
        logging.info(f"Found {len(bboxes)} cores.")
        
        logging.info("Extracting patches...")
        extract_patches(args.he_slide, args.orion_slide, bboxes, args.output_dir, args.target_size)
        
    elif args.command == "register":
        logging.info("Starting registration...")
        register_slides(args.he_slide, args.orion_slide, args.output_dir)
        logging.info("Registration complete.")
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
