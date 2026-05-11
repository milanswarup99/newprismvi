#!/usr/bin/env python3
"""
DNG Raw Image Analyzer - Local Version
Adapted from PrismVI25.ipynb for local execution
"""

import os
import glob
import sys
from pathlib import Path
import joblib
from skimage.color import rgb2lab, deltaE_ciede2000

try:
    model = joblib.load(
        "saturation_model.pkl"
    )
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['rawpy', 'exifread', 'pandas', 'numpy', 'cv2', 'matplotlib', 'scikit-image', 'scikit-learn', 'joblib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def find_dng_files():
    """Find DNG files in current workspace"""
    search_paths = [
        ".",  # Current directory
        "./dng_files",  # Local dng_files directory
        "./sample_dng",  # Sample directory
    ]
    
    dng_files = []
    for path_pattern in search_paths:
        if "**" in path_pattern:
            # Recursive search
            base_path = path_pattern.split("**")[0]
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.lower().endswith('.dng'):
                        dng_files.append(os.path.join(root, file))
        else:
            # Direct pattern search
            pattern = os.path.join(path_pattern, "*.dng")
            dng_files.extend(glob.glob(pattern))
    
    return dng_files
    
def predict_saturation_level(rgb16, raw):
    """Predict saturation level using ML model"""

    if model is None:
        return -1
    # Normalize image
    rgb_norm = rgb16.astype(np.float32) / 65535.0
    # Convert to LAB
    lab = rgb2lab(rgb_norm)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    # Chroma
    chroma = np.sqrt(a**2 + b**2)
    mean_chroma = np.mean(chroma)
    # Slightly boosted reference image
    rgb_ref = np.clip(rgb_norm * 1.05, 0, 1)
    lab_ref = rgb2lab(rgb_ref)

    # Delta E
    deltaE = np.mean(deltaE_ciede2000(lab, lab_ref))

    # Safe white balance extraction
    wb = list(raw.camera_whitebalance)

    while len(wb) < 4:
        wb.append(0)

    # Safe black level extraction
    bl = list(raw.black_level_per_channel)

    while len(bl) < 4:
        bl.append(0)

    feature_vector = [[
        bl[0],
        bl[1],
        bl[2],
        bl[3],
        raw.white_level,
        wb[0],
        wb[1],
        wb[2],
        wb[3],
        mean_chroma,
        deltaE
    ]]

    predicted_level = model.predict(feature_vector)[0]
    return int(predicted_level)

def analyze_single_dng(dng_path):
    """Analyze a single DNG file"""
    import exifread
    import rawpy
    import pandas as pd
    import cv2
    import numpy as np
    
    print(f"\n=== Analyzing: {os.path.basename(dng_path)} ===")
    
    metadata = {}
    
    # Extract EXIF metadata
    try:
        with open(dng_path, "rb") as f:
            tags = exifread.process_file(f, details=True)
        
        print(f"EXIF Tags Found: {len(tags)}")
        
        # Print key camera info
        key_tags = ['Image Make', 'Image Model', 'EXIF ISOSpeedRatings', 
                   'EXIF FNumber', 'EXIF ExposureTime', 'EXIF DateTimeOriginal']
        
        for tag in key_tags:
            if tag in tags:
                print(f"  {tag}: {tags[tag]}")
                metadata[f"EXIF_{tag}"] = str(tags[tag])
        
    except Exception as e:
        print(f"Error reading EXIF: {e}")
    
    # Extract rawpy metadata and process image
    try:
        with rawpy.imread(dng_path) as raw:
            print(f"Raw Image Size: {raw.sizes.raw_width}x{raw.sizes.raw_height}")
            print(f"Visible Size: {raw.sizes.width}x{raw.sizes.height}")
            print(f"Color Pattern: {raw.color_desc}")
            print(f"White Level: {raw.white_level}")
            print(f"Black Levels: {raw.black_level_per_channel}")
            
            # Store rawpy metadata
            for attr in dir(raw):
                if not attr.startswith("_") and attr not in ['color_desc', 'raw_colors', 
                                                           'raw_colors_visible', 'raw_image', 
                                                           'raw_image_visible']:
                    try:
                        value = getattr(raw, attr)
                        if not callable(value):
                            metadata[f"RAWPY_{attr}"] = str(value)
                    except Exception:
                        continue
            
            # Process image for saturation analysis
            rgb16 = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)
            rgb8 = (rgb16 / 256).astype("uint8")
            
            # Convert to HSV for saturation analysis
            bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV)
            
            # Calculate saturation metrics
            S = hsv[:, :, 1].astype(np.float32)
            mean_sat = float(S.mean() / 255.0)
            max_sat = float(S.max() / 255.0)
            min_sat = float(S.min() / 255.0)
            
            print(f"Saturation - Mean: {mean_sat:.4f}, Max: {max_sat:.4f}, Min: {min_sat:.4f}")
            
            metadata['SATURATION_MEAN'] = mean_sat
            metadata['SATURATION_MAX'] = max_sat
            metadata['SATURATION_MIN'] = min_sat
            
            # Add ML prediction
            predicted_level = predict_saturation_level(
                rgb16,
                raw
            )

            print(f"ML Predicted Level: {predicted_level}")

            metadata['ML_PREDICTED_LEVEL'] = predicted_level
    except Exception as e:
        print(f"Error processing raw image: {e}")
        return None
    
    return metadata

def batch_analyze_dng_files(dng_files):
    """Analyze multiple DNG files and create summary"""
    import pandas as pd
    
    print(f"\n=== Batch Analysis of {len(dng_files)} DNG files ===")
    
    all_metadata = []
    saturation_results = []
    
    for i, dng_path in enumerate(dng_files):
        print(f"\n[{i+1}/{len(dng_files)}] Processing: {os.path.basename(dng_path)}")
        
        metadata = analyze_single_dng(dng_path)
        if metadata:
            all_metadata.append({
                'filename': os.path.basename(dng_path),
                'filepath': dng_path,
                **metadata
            })
            
            if 'SATURATION_MEAN' in metadata:
                saturation_results.append((os.path.basename(dng_path), metadata['SATURATION_MEAN']))
    
    # Create DataFrames
    if all_metadata:
        df_metadata = pd.DataFrame(all_metadata)
        
        # Save metadata to CSV
        output_path = "dng_metadata.csv"
        df_metadata.to_csv(output_path, index=False)
        print(f"\nMetadata saved to: {output_path}")
        
        # Saturation ranking
        if saturation_results:
            saturation_results.sort(key=lambda x: x[1])
            print("\n=== Images Ranked by Saturation (Low to High) ===")
            for filename, sat in saturation_results:
                print(f"{sat:.4f}  {filename}")
        
        return df_metadata
    else:
        print("No metadata collected.")
        return None

def main():
    """Main execution function"""
    print("=== DNG Raw Image Analyzer ===")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Find DNG files
    print("\nSearching for DNG files...")
    dng_files = find_dng_files()
    
    if not dng_files:
        print("No DNG files found!")
        print("Please ensure DNG files are in current directory or subdirectories.")
        print("Supported locations:")
        print("  - Current directory (./)")
        print("  - ./dng_files/")
        print("  - ./sample_dng/")
        return
    
    print(f"Found {len(dng_files)} DNG files")
    
    # Interactive menu
    while True:
        print("\n=== Options ===")
        print("1. Analyze a single DNG file")
        print("2. Batch analyze all DNG files")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\nAvailable DNG files:")
            for i, file in enumerate(dng_files[:10]):  # Show first 10
                print(f"{i+1}. {os.path.basename(file)}")
            
            if len(dng_files) > 10:
                print(f"... and {len(dng_files) - 10} more files")
            
            try:
                file_idx = int(input(f"\nEnter file number (1-{len(dng_files)}: ")) - 1
                if 0 <= file_idx < len(dng_files):
                    analyze_single_dng(dng_files[file_idx])
                else:
                    print("Invalid file number.")
            except ValueError:
                print("Invalid input.")
        
        elif choice == '2':
            batch_analyze_dng_files(dng_files)
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-3.")

if __name__ == "__main__":
    main()
