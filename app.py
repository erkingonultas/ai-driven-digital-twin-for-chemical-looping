import numpy as np
import gpytorch
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt

# Simulated Chemical Looping Process Function
def chemical_looping_process(T, OC):
    """
    Simulates hydrogen yield and safety factor based on reactor temperature (T) and oxygen carrier efficiency (OC).
    """
    H2_yield = np.exp(-((T - 850) / 50) ** 2) * OC  # Gaussian peak at optimal temperature
    SF = 1 - np.exp(-((T - 900) / 100) ** 2)  # Higher T increases risk (Safety Factor closer to 0 is safer)
    return H2_yield, SF

# Generate Initial Training Data
T_samples = np.random.uniform(700, 1000, 10)  # Reactor Temperature (K)
OC_samples = np.random.uniform(0.7, 1.0, 10)  # Oxygen Carrier Efficiency

H2_yield_samples, SF_samples = zip(*[chemical_looping_process(T, OC) for T, OC in zip(T_samples, OC_samples)])
H2_yield_samples, SF_samples = np.array(H2_yield_samples), np.array(SF_samples)

# Train Gaussian Process Model
def train_gp(X, Y):
    """Train a Gaussian Process model."""
    # Convert inputs to double precision
    X_tensor = torch.tensor(X, dtype=torch.float64)
    Y_tensor = torch.tensor(Y, dtype=torch.float64).unsqueeze(-1)

    # Standardize the input data
    epsilon = 1e-8
    X_mean = X_tensor.mean(dim=0)
    X_std = X_tensor.std(dim=0) + epsilon
    X_tensor_standardized = (X_tensor - X_mean) / X_std

    # Min-Max Scaling
    X_min = X_tensor_standardized.min(dim=0).values
    X_max = X_tensor_standardized.max(dim=0).values
    X_tensor_normalized = (X_tensor - X_min) / (X_max - X_min)

    gp_model = SingleTaskGP(X_tensor_normalized, Y_tensor)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_model(mll)

    return gp_model

# Convert inputs into feature vectors
X_train = np.column_stack((T_samples, OC_samples))
Y_train = H2_yield_samples

# Train model
gp_model = train_gp(X_train, Y_train)

# Bayesian Optimization: Find Optimal T and OC
def optimize_process():
    """Use Bayesian Optimization to find optimal process parameters."""
    bounds = torch.tensor([[700.0, 0.7], [1000.0, 1.0]])  # Bounds for T and OC
    
    EI = ExpectedImprovement(gp_model, best_f=Y_train.max(), maximize=True)
    
    candidates, _ = optimize_acqf(EI, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
    
    return candidates.detach().numpy().flatten()

optimal_T, optimal_OC = optimize_process()
optimal_H2, optimal_SF = chemical_looping_process(optimal_T, optimal_OC)

# Fault Detection: Anomaly Detection Based on Expected Yield Range
def detect_fault(T, OC, threshold=0.2):
    """Detect anomalies by comparing predicted yield with actual yield."""
    predicted_H2, _ = chemical_looping_process(T, OC)
    actual_H2 = np.random.normal(predicted_H2, 0.05) # Simulated real-world data with noise
    deviation =  abs(predicted_H2 - actual_H2)

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
