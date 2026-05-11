# -*- coding: utf-8 -*-
"""ML Model for Saturation Prediction
Trained on DNG dataset to predict optimal saturation enhancement levels (1-10)
"""

import os
import rawpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, deltaE_ciede2000
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_saturation_model(dataset_path=None):
    """
    Train saturation prediction model on DNG dataset
    Returns trained model and accuracy metrics
    """
    
    # If no dataset path provided, try to download from HuggingFace
    if dataset_path is None:
        try:
            from huggingface_hub import snapshot_download
            dataset_path = snapshot_download(
                repo_id="milan9999/dng_50",
                repo_type="dataset"
            )
            print("Dataset downloaded to:", dataset_path)
        except Exception as e:
            print(f"Could not download dataset: {e}")
            print("Please provide local dataset path")
            return None, None
    
    # Get all DNG files
    dng_files = [f for f in os.listdir(dataset_path) if f.endswith(".dng")]
    print(f"Found {len(dng_files)} DNG files")
    
    if len(dng_files) == 0:
        print("No DNG files found in dataset")
        return None, None
    
    # Process all images
    results = []
    
    for file in dng_files:
        try:
            path = os.path.join(dataset_path, file)
            
            with rawpy.imread(path) as raw:
                # Raw to RGB
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=16
                )
                
                # Resize for faster processing
                rgb = cv2.resize(rgb, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                
                # Normalize
                rgb_norm = rgb / 65535.0
                
                # RGB to LAB
                lab = rgb2lab(rgb_norm)
                
                # Calculate chroma
                a = lab[:, :, 1]
                b = lab[:, :, 2]
                chroma = np.sqrt(a**2 + b**2)
                mean_chroma = np.mean(chroma)
                
                # Calculate Delta E (baseline)
                rgb_ref = np.clip(rgb * 1.05, 0, 65535)
                lab_ref = rgb2lab(rgb_ref / 65535.0)
                deltaE = np.mean(deltaE_ciede2000(lab, lab_ref))
                
                # Extract metadata
                row = {
                    "image_name": file,
                    "mean_chroma": mean_chroma,
                    "deltaE": deltaE,
                    "black_level_r": raw.black_level_per_channel[0],
                    "black_level_g1": raw.black_level_per_channel[1],
                    "black_level_g2": raw.black_level_per_channel[2],
                    "black_level_b": raw.black_level_per_channel[3],
                    "white_level": raw.white_level,
                    "wb_r": raw.camera_whitebalance[0],
                    "wb_g1": raw.camera_whitebalance[1],
                    "wb_g2": raw.camera_whitebalance[2],
                    "wb_b": raw.camera_whitebalance[3]
                }
                
                results.append(row)
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    if not results:
        print("No images processed successfully")
        return None, None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Create saturation levels (1-10 quantiles)
    df["saturation_level"] = pd.qcut(
        df["mean_chroma"],
        10,
        labels=[1,2,3,4,5,6,7,8,9,10]
    )
    
    # Save dataset
    df.to_csv("final_milestone1_dataset.csv", index=False)
    print("Dataset saved to final_milestone1_dataset.csv")
    
    # Prepare data for ML
    X = df.drop(columns=["image_name", "saturation_level"])
    y = df["saturation_level"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Feature importance
    feature_importance = pd.Series(
        rf_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    
    print("Feature Importance:")
    print(feature_importance)
    
    # Save model
    joblib.dump(rf_model, "saturation_model.pkl")
    print("Model saved as saturation_model.pkl")
    
    return rf_model, rf_accuracy

def predict_saturation_level(rgb16, raw, model_path="saturation_model.pkl"):
    """
    Predict saturation level using trained ML model
    
    Args:
        rgb16: 16-bit RGB image array
        raw: rawpy image object
        model_path: path to saved model file
    
    Returns:
        int: predicted saturation level (1-10), or -1 if model not available
    """
    try:
        # Load model
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return -1
    
    # Normalize image
    rgb_norm = rgb16.astype(np.float32) / 65535.0
    
    # Convert to LAB
    lab = rgb2lab(rgb_norm)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
    # Calculate chroma
    chroma = np.sqrt(a**2 + b**2)
    mean_chroma = np.mean(chroma)
    
    # Calculate Delta E
    rgb_ref = np.clip(rgb_norm * 1.05, 0, 1)
    lab_ref = rgb2lab(rgb_ref)
    deltaE = np.mean(deltaE_ciede2000(lab, lab_ref))
    
    # Safe white balance extraction
    wb = list(raw.camera_whitebalance)
    while len(wb) < 4:
        wb.append(0)
    
    # Safe black level extraction
    bl = list(raw.black_level_per_channel)
    while len(bl) < 4:
        bl.append(0)
    
    # Create feature vector
    feature_vector = [[
        bl[0], bl[1], bl[2], bl[3],  # Black levels
        raw.white_level,                 # White level
        wb[0], wb[1], wb[2], wb[3],   # White balance
        mean_chroma,                    # Mean chroma
        deltaE                          # Delta E
    ]]
    
    # Predict
    predicted_level = model.predict(feature_vector)[0]
    return int(predicted_level)

if __name__ == "__main__":
    # Train model if run directly
    model, accuracy = train_saturation_model()
    if model:
        print(f"Model trained successfully with accuracy: {accuracy:.4f}")
    else:
        print("Model training failed")
