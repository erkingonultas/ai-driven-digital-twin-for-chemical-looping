import numpy as np
import gpytorch
import torch
import sqlite3
import pandas as pd
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt

# Load Data from Database
conn = sqlite3.connect("chemical_looping.db")
df = pd.read_sql("SELECT * FROM chemical_data", conn)
conn.close()

# Extract Data
T_samples = df['Temperature_K'].values
OC_samples = df['Oxygen_Carrier_Efficiency'].values
H2_yield_samples = df['Hydrogen_Yield'].values
SF_samples = df['Safety_Factor'].values

# Train Gaussian Process Model
def train_gp(X, Y):
    """Train a Gaussian Process model."""
    X_tensor = torch.tensor(X, dtype=torch.float64)
    Y_tensor = torch.tensor(Y, dtype=torch.float64).unsqueeze(-1)
    
    likelihood = GaussianLikelihood()
    gp_model = SingleTaskGP(X_tensor, Y_tensor, likelihood=likelihood)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_model(mll)
    
    return gp_model

# Convert inputs into feature vectors
X_train = np.column_stack((T_samples, OC_samples))
Y_train = H2_yield_samples

# Train GP model
gp_model = train_gp(X_train, Y_train)

# Bayesian Optimization: Find Optimal T and OC
def optimize_process():
    """Use Bayesian Optimization to find optimal process parameters."""
    bounds = torch.tensor([[700.0, 0.7], [1000.0, 1.0]])  # Bounds for T and OC
    
    EI = ExpectedImprovement(gp_model, best_f=Y_train.max(), maximize=True)
    
    candidates, _ = optimize_acqf(EI, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
    
    return candidates.detach().numpy().flatten()

optimal_T, optimal_OC = optimize_process()

# Simulated Chemical Looping Process Function
def chemical_looping_process(T, OC):
    """
    Simulates hydrogen yield and safety factor based on reactor temperature (T) and oxygen carrier efficiency (OC).
    """
    H2_yield = np.exp(-((T - 850) / 50) ** 2) * OC  # Gaussian peak at optimal temperature
    SF = 1 - np.exp(-((T - 900) / 100) ** 2)  # Higher T increases risk (Closer to 0 is safer)
    return H2_yield, SF

optimal_H2, optimal_SF = chemical_looping_process(optimal_T, optimal_OC)

# Fault Detection: Anomaly Detection Based on Expected Yield Range
def detect_fault(T, OC, threshold=0.2):
    """Detect anomalies by comparing predicted yield with actual yield."""
    predicted_H2, _ = chemical_looping_process(T, OC)
    actual_H2 = np.random.normal(predicted_H2, 0.05) # Simulated real-world data with noise
    deviation = abs(predicted_H2 - actual_H2)
    
    if deviation > threshold:
        print(f"\n⚠️ FAULT DETECTED! Large deviation in hydrogen yield: {deviation:.4f}")
    else:
        print("\n✅ System Operating Normally.")
    return deviation

# Run Fault Detection
deviation = detect_fault(optimal_T, optimal_OC)

# Display Results
print("\n===========================================")
print("        OPTIMIZATION RESULTS")
print("===========================================")
print(f"Optimal Reactor Temperature:       {optimal_T:.2f} K")
print(f"Optimal Oxygen Carrier Efficiency: {optimal_OC:.2f}")
print(f"Predicted Hydrogen Yield:          {optimal_H2:.4f}")
print(f"Process Safety Factor:             {optimal_SF:.4f} (Closer to 0 is safer)")
print(f"Fault Detection Deviation:         {deviation:.4f}")
print("===========================================\n")

# Visualize Results
plt.figure(figsize=(10, 6))
plt.scatter(T_samples, H2_yield_samples, color='blue', label='Training Data')
plt.plot([optimal_T], [optimal_H2], 'ro', markersize=15, label='Optimal Point')
plt.xlabel('Temperature (K)')
plt.ylabel('Hydrogen Yield')
plt.title('Optimization Results')
plt.legend()
plt.show()
