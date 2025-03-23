import numpy as np
import pandas as pd
import sqlite3

# Simulated Chemical Looping Process Function
def chemical_looping_process(T, OC):
    """
    Simulates hydrogen yield and safety factor based on reactor temperature (T) and oxygen carrier efficiency (OC).
    """
    H2_yield = np.exp(-((T - 850) / 50) ** 2) * OC  # Gaussian peak at optimal temperature
    SF = 1 - np.exp(-((T - 900) / 100) ** 2)  # Higher T increases risk (Closer to 0 is safer)
    return H2_yield, SF

# Generate Synthetic Data
num_samples = 10000
T_samples = np.random.uniform(700, 1000, num_samples)  # Reactor Temperature (K)
OC_samples = np.random.uniform(0.7, 1.0, num_samples)  # Oxygen Carrier Efficiency

H2_yield_samples, SF_samples = zip(*[chemical_looping_process(T, OC) for T, OC in zip(T_samples, OC_samples)])
H2_yield_samples, SF_samples = np.array(H2_yield_samples), np.array(SF_samples)

# Introduce Fault Labels (1: Unsafe, 0: Safe)
fault_labels = (SF_samples > 0.5).astype(int)  # Unsafe if SF > 0.5

# Create DataFrame
df = pd.DataFrame({
    'Temperature_K': T_samples,
    'Oxygen_Carrier_Efficiency': OC_samples,
    'Hydrogen_Yield': H2_yield_samples,
    'Safety_Factor': SF_samples,
    'Fault_Label': fault_labels
})

# Save to SQLite Database
conn = sqlite3.connect("chemical_looping.db")
df.to_sql("chemical_data", conn, if_exists="replace", index=False)
conn.close()

print("Database successfully created with 10,000 samples!")