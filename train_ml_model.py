#!/usr/bin/env python3
"""
Train ML Model for Saturation Prediction
Based on the working notebook code
"""

import os
import rawpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, deltaE_ciede2000
import cv2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def download_dataset():
    """Download dataset from HuggingFace"""
    try:
        from huggingface_hub import snapshot_download
        dataset_path = snapshot_download(
            repo_id="milan9999/dng_50",
            repo_type="dataset"
        )
        print("Dataset downloaded to:", dataset_path)
        return dataset_path
    except Exception as e:
        print(f"Could not download dataset: {e}")
        return None

def train_model(dataset_path=None):
    """Train saturation prediction model on DNG dataset"""
    
    if dataset_path is None:
        dataset_path = download_dataset()
    
    if dataset_path is None:
        print("No dataset available")
        return None, None
    
    # Remove any existing demo data
    if os.path.exists("final_milestone1_dataset.csv"):
        os.remove("final_milestone1_dataset.csv")
        print("Removed existing demo dataset")
    
    # Remove any existing model
    if os.path.exists("saturation_model.pkl"):
        os.remove("saturation_model.pkl")
        print("Removed existing model")
    
    # Get all DNG files
    dng_files = [f for f in os.listdir(dataset_path) if f.endswith(".dng")]
    print(f"Found {len(dng_files)} DNG files")
    
    if len(dng_files) == 0:
        print("No DNG files found")
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
    
    # Create saturation levels (1-5 quantiles for better distribution)
    df["saturation_level"] = pd.qcut(
        df["mean_chroma"],
        5,
        labels=[1,2,3,4,5]
    )
    
    # Save dataset
    df.to_csv("final_milestone1_dataset.csv", index=False)
    print("Dataset saved to final_milestone1_dataset.csv")
    
    # Prepare data for ML
    X = df.drop(columns=["image_name", "saturation_level"])
    y = df["saturation_level"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Decision Tree model (100% accuracy in notebook)
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Evaluate Decision Tree
    dt_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate Random Forest
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
    
    # Save the better model (Decision Tree had 100% accuracy)
    best_model = dt_model if dt_accuracy > rf_accuracy else rf_model
    joblib.dump(best_model, "saturation_model.pkl")
    print("Model saved as saturation_model.pkl")
    
    return best_model, dt_accuracy if dt_accuracy > rf_accuracy else rf_accuracy

def predict_saturation_level(rgb16, raw, model_path="saturation_model.pkl"):
    """Predict saturation level using trained ML model"""
    try:
        # Load model
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 5
    
    try:
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
        
        # Create feature vector (same as notebook)
        feature_vector = [[
            bl[0], bl[1], bl[2], bl[3],  # Black levels
            raw.white_level,                 # White level
            wb[0], wb[1], wb[2], wb[3],   # White balance
            mean_chroma,                    # Mean chroma
            deltaE                          # Delta E
        ]]
        
        # Predict
        predicted_level = model.predict(feature_vector)[0]
        print(f"DEBUG: Raw ML prediction: {predicted_level}")
        
        # Map 1-5 prediction to 5-10 range (reverse mapping - lower saturation = higher enhancement)
        if predicted_level == 1:  # Very low saturation
            mapped_level = 9  # Strong enhancement
        elif predicted_level == 2:  # Low saturation
            mapped_level = 8  # Good enhancement
        elif predicted_level == 3:  # Medium saturation
            mapped_level = 7  # Moderate enhancement
        elif predicted_level == 4:  # High saturation
            mapped_level = 6  # Mild enhancement
        elif predicted_level == 5:  # Very high saturation
            mapped_level = 5  # Minimal enhancement
        else:
            mapped_level = 5  # Default
            
        print(f"DEBUG: Mapped level: {mapped_level}")
        return mapped_level
        
    except Exception as e:
        print(f"Error predicting saturation: {e}")
        return 5

if __name__ == "__main__":
    # Train model if run directly
    model, accuracy = train_model()
    if model:
        print(f"Model trained successfully with accuracy: {accuracy:.4f}")
    else:
        print("Model training failed")
