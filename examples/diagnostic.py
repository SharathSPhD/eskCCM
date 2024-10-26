import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skccm import CCM, Embed
import os
from modules import ExtendedCCM

def plot_ccm_skills(ccm_skill, title, test_name, results_dir='results'):
    """Plot cross-map skills vs lags and save to file."""
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    lags = ccm_skill[:, 0]
    skill_x_to_y = ccm_skill[:, 1]
    skill_y_to_x = ccm_skill[:, 2]
    
    plt.plot(lags, skill_x_to_y, 'b-o', label='X → Y', alpha=0.7)
    plt.plot(lags, skill_y_to_x, 'r-o', label='Y → X', alpha=0.7)
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    plt.xlabel('Cross-Map Lag')
    plt.ylabel('Cross-Map Skill (ρ)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = os.path.join(results_dir, f'{test_name}_ccm_skills.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_series(x, y, title, test_name, results_dir='results'):
    """Plot time series data."""
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    t = np.arange(len(x))
    plt.plot(t, x, 'b-', label='X', alpha=0.7)
    plt.plot(t, y, 'r-', label='Y', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = os.path.join(results_dir, f'{test_name}_time_series.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def generate_bidirectional_data(ts_length=3000, tau_d=2):
    """Generate bidirectional coupled system following Sugihara's equations."""
    rx = 3.8
    ry = 3.5
    Axy = 0.07
    Ayx = 0.08
    
    x = np.zeros(ts_length)
    y = np.zeros(ts_length)
    
    x[0] = 0.4
    y[0] = 0.2
    
    for t in range(tau_d):
        x[t + 1] = x[t] * (rx - rx * x[t] - Axy * y[t])
        y[t + 1] = y[t] * (ry - ry * y[t])
    
    for t in range(tau_d, ts_length - 1):
        x[t + 1] = x[t] * (rx - rx * x[t] - Axy * y[t])
        y[t + 1] = y[t] * (ry - ry * y[t] - Ayx * x[t - tau_d])
        
        x[t + 1] = np.clip(x[t + 1], 0, 1)
        y[t + 1] = np.clip(y[t + 1], 0, 1)
    
    return x, y

def generate_synchrony_data(ts_length=3000):
    """Generate synchrony data following Eq. S2."""
    rx = 3.8
    ry = 3.1
    Ayx = 0.9
    
    x = np.zeros(ts_length)
    y = np.zeros(ts_length)
    
    x[0] = 0.4
    y[0] = 0.2
    
    for t in range(ts_length - 1):
        x[t + 1] = x[t] * (rx - rx * x[t])
        y[t + 1] = y[t] * (ry - ry * y[t] - Ayx * x[t])
        
        x[t + 1] = np.clip(x[t + 1], 0, 1)
        y[t + 1] = np.clip(y[t + 1], 0, 1)
    
    return x, y

def generate_transitive_data(ts_length=3000):
    """Generate transitive causality data with stronger coupling."""
    r1 = 3.9
    r2 = 3.6
    r3 = 3.6
    r4 = 3.8
    
    # Stronger coupling coefficients
    A21 = 0.45
    A32 = 0.45
    A43 = 0.45
    
    y1 = np.zeros(ts_length)
    y2 = np.zeros(ts_length)
    y3 = np.zeros(ts_length)
    y4 = np.zeros(ts_length)
    
    # Set more distinct initial conditions
    y1[0] = 0.4
    y2[0] = 0.3
    y3[0] = 0.2
    y4[0] = 0.1
    
    for t in range(ts_length - 1):
        y1[t + 1] = y1[t] * (r1 - r1 * y1[t])
        y2[t + 1] = y2[t] * (r2 - r2 * y2[t] - A21 * y1[t])
        y3[t + 1] = y3[t] * (r3 - r3 * y3[t] - A32 * y2[t])
        y4[t + 1] = y4[t] * (r4 - r4 * y4[t] - A43 * y3[t])
        
        y1[t + 1] = np.clip(y1[t + 1], 0, 1)
        y2[t + 1] = np.clip(y2[t + 1], 0, 1)
        y3[t + 1] = np.clip(y3[t + 1], 0, 1)
        y4[t + 1] = np.clip(y4[t + 1], 0, 1)
    
    return y1, y2, y3, y4

def test_ccm_analysis(causality_type, library_length=None):
    """Run CCM analysis for specified causality type."""
    # Generate appropriate data
    if causality_type == 'synchrony':
        x, y = generate_synchrony_data()
        embed_dim = 2
        lib_len = 200 if library_length is None else library_length
    elif causality_type == 'bidirectional':
        x, y = generate_bidirectional_data()
        embed_dim = 3
        lib_len = 200 if library_length is None else library_length
    else:  # transitive
        y1, y2, y3, y4 = generate_transitive_data()
        x, y = y1, y4
        embed_dim = 4
        lib_len = 400 if library_length is None else library_length
    
    # Get stable portion of time series
    if causality_type == 'transitive':
        x_stable = x[200:2200]
        y_stable = y[200:2200]
    else:
        x_stable = x[100:2000]
        y_stable = y[100:2000]
    
    # Plot time series
    plot_time_series(x_stable, y_stable, 
                    f"{causality_type.capitalize()}: Time Series",
                    f"{causality_type.lower()}_raw")
    
    # Create embeddings
    embed_x = Embed(x_stable)
    embed_y = Embed(y_stable)
    
    # Test range of lags
    lags = np.arange(-8, 9)
    ccm_skill = []
    
    print(f"\nTesting {causality_type} causality...")
    
    for lag in lags:
        # Create time-lagged embeddings
        X1 = embed_x.embed_vectors_1d(lag=abs(lag) if lag != 0 else 1, embed=embed_dim)
        X2 = embed_y.embed_vectors_1d(lag=abs(lag) if lag != 0 else 1, embed=embed_dim)
        
        # Adjust for positive/negative lags
        if lag > 0:
            X1, X2 = X1[:-lag], X2[lag:]
        elif lag < 0:
            X1, X2 = X1[-lag:], X2[:lag]
        
        # Ensure equal lengths
        min_len = min(len(X1), len(X2))
        X1, X2 = X1[:min_len], X2[:min_len]
        
        # Initialize CCM
        ccm = CCM()
        
        # Split into library and prediction sets
        X1_lib, X2_lib = X1[:lib_len], X2[:lib_len]
        X1_pred, X2_pred = X1[lib_len:], X2[lib_len:]
        
        # Fit and predict
        ccm.fit(X1_lib, X2_lib)
        _, _ = ccm.predict(X1_pred, X2_pred, [lib_len])
        score_x, score_y = ccm.score()
        
        print(f"\nLag {lag}:")
        print(f"Embedding shapes - X1: {X1.shape}, X2: {X2.shape}")
        print(f"Library shapes - X1_lib: {X1_lib.shape}, X2_lib: {X2_lib.shape}")
        print(f"X→Y score: {score_x[0]:.3f}")
        print(f"Y→X score: {score_y[0]:.3f}")
        
        ccm_skill.append((lag, score_x[0], score_y[0]))
    
    ccm_skill = np.array(ccm_skill)
    plot_ccm_skills(ccm_skill, f"{causality_type.capitalize()}: CCM Skills vs Lag", 
                   f"{causality_type.lower()}_ccm_test")
    
    return ccm_skill

def test_extended_ccm():
    """Test the ExtendedCCM class."""
    # Generate bidirectional data
    x, y = generate_bidirectional_data()
    extended_ccm = ExtendedCCM(x, y, max_dim=3)
    lags = np.arange(-8, 9)
    
    x_stable = x[100:2000]
    y_stable = y[100:2000]
    
    causality_type, ccm_skill = extended_ccm.detect_causality(
        x_stable, y_stable,
        lags,
        embed_dim=2,
        library_length=500
    )
    
    plot_ccm_skills(ccm_skill, "ExtendedCCM: Bidirectional Causality", 
                   "extended_ccm_bidirectional", "results")
    
    print(f"Detected causality type: {causality_type}")

if __name__ == "__main__":
    # Test all causality types
    for causality_type in ['synchrony', 'bidirectional', 'transitive']:
        ccm_skill = test_ccm_analysis(causality_type)
    
    # Test the ExtendedCCM class
    test_extended_ccm()
    
    print("\nResults saved to 'results' directory.")
