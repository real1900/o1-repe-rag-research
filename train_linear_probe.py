import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Phase 3: Linear Probe Training (The Predictor)
# Goal: Train a lightweight linear model to predict the Optimal Alpha from prompt geometry

def train_probe(csv_path="synthetic_alpha_tuning.csv", model_path="linear_probe_weights.pt", scaler_path="feature_scaler.pkl"):
    print(f"Loading synthetic data from {csv_path}...")
    
    if not os.path.exists(csv_path):
        print("Data generation is still running. Please wait for synthetic_alpha_tuning.csv to be completely generated.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Total rows in dataset: {len(df)}")
    
    # We only want to train on successful discoveries where the model was successfully steered
    df_success = df[(df['Success'] == True) & (df['Optimal_Alpha'] > 0.0)].copy()
    print(f"Successful Alpha Discoveries available for training: {len(df_success)}")
    
    if len(df_success) < 50:
        print("Not enough successful rows to train a stable model yet. Let the generator run longer.")
        return

    # Define the 7 Input Features exactly as extracted in the generation script
    # We include Collapse_Alpha as a feature because the optimal alpha is usually a direct fraction of it
    feature_cols = [
        "Prompt_Norm", 
        "Concept_Norm", 
        "Dot_Product", 
        "Cosine_Sim", 
        "Token_Confidence", 
        "Prompt_Length", 
        "Collapse_Alpha"
    ]
    
    X = df_success[feature_cols].values
    y = df_success["Optimal_Alpha"].values.reshape(-1, 1) # Target variable
    
    # Split into Train and Validation sets (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardization is strictly required for Linear Regression with large geometric dots/norms
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    print("\nInitializing O(1) Non-Linear MLP Probe Architecture...")
    # The architecture: A tiny 2-layer MLP to map complex geometric bounds
    class AlphaPredictor(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            
        def forward(self, x):
            return self.net(x)
            
    model = AlphaPredictor(input_dim=len(feature_cols))
    
    # Training Loop setup
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss() # Used for intuitive error reporting (Mean Absolute Error)
    
    # L2 Regularization (weight_decay) added to prevent overfitting geometric features
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    epochs = 1500
    print("Beginning Training Loop...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 300 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t)
                val_loss = criterion(val_preds, y_val_t)
                val_mae = mae_metric(val_preds, y_val_t)
            print(f"Epoch [{epoch+1}/{epochs}] | Train MSE: {loss.item():.5f} | Val MSE: {val_loss.item():.5f} | Val MAE (Avg Error): {val_mae.item():.5f}")
            
    print("\nTraining Complete!")
    
    # Save the architecture and scaler to disk for Phase 4 inference
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Saved Linear Probe structural weights to {model_path}")
    print(f"Saved Input Standardization Scaler to {scaler_path}")
    
    # Output some test predictions to verify it learned the bounds
    print("\n--- Validation Sample Predictions ---")
    model.eval()
    with torch.no_grad():
        sample_preds = model(X_val_t[:5])
        
    for i in range(5):
        actual = y_val_t[i].item()
        predicted = sample_preds[i].item()
        diff = abs(actual - predicted)
        print(f"Actual Optimal Alpha: {actual:.5f} | Predicted by Probe: {predicted:.5f} | Error: {diff:.5f}")

if __name__ == "__main__":
    train_probe()
