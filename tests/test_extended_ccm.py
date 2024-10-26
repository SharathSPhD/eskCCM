import numpy as np
import pytest
from eskCCM import CCM, Embed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from eskCCM.extended_ccm import ExtendedCCM
from examples.diagnostic import test_extended_ccm

# Helper visualization functions for debugging
def plot_ccm_skills(ccm_skill, title, test_name, results_dir='test_results'):
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

# Data generation functions
def generate_bidirectional_data(ts_length=3000, tau_d=2):
    """Generate bidirectional coupled system following Sugihara's equations."""
    rx = 3.78  # Fixed parameter from paper
    ry = 3.78  # Fixed parameter from paper
    Axy = 0.07  # Coupling strength
    Ayx = 0.08  # Coupling strength
    
    x = np.zeros(ts_length)
    y = np.zeros(ts_length)
    
    x[0] = 0.2
    y[0] = 0.4
    
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
    Ayx = 0.9  # Strong unidirectional coupling
    
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
    
    A21 = 0.45
    A32 = 0.45
    A43 = 0.45
    
    y1 = np.zeros(ts_length)
    y2 = np.zeros(ts_length)
    y3 = np.zeros(ts_length)
    y4 = np.zeros(ts_length)
    
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

# Test functions
def test_extended_ccm_inheritance():
    """Test that ExtendedCCM properly inherits from CCM."""
    x = np.random.rand(100)
    y = np.random.rand(100)
    
    extended_ccm = ExtendedCCM(x, y)
    assert isinstance(extended_ccm, CCM), "ExtendedCCM should inherit from CCM"

def test_embedding():
    """Test embedding functionality with simple sequence."""
    ts_length = 100
    x = np.arange(ts_length, dtype=float)
    y = np.arange(ts_length, dtype=float)
    
    extended_ccm = ExtendedCCM(x, y)
    
    # Test positive lag
    lag = 2
    embed_dim = 3
    embedding = extended_ccm.embed_with_lag(x, lag, embed_dim)
    
    assert embedding.shape[1] == embed_dim, "Embedding dimension should match specified dimension"
    assert len(embedding) == ts_length - lag * (embed_dim - 1), "Embedding length should be correct"
    
    # Test negative lag
    lag = -2
    embedding = extended_ccm.embed_with_lag(x, lag, embed_dim)
    assert embedding.shape[1] == embed_dim, "Embedding dimension should match for negative lag"

def test_invalid_inputs():
    """Test handling of invalid inputs."""
    with pytest.raises(ValueError):
        extended_ccm = ExtendedCCM(np.array([]), np.array([]))
    
    x = np.random.rand(100)
    y = np.random.rand(100)
    extended_ccm = ExtendedCCM(x, y)
    
    with pytest.raises(ValueError):
        extended_ccm.embed_with_lag(x, 1000, 2)
    
    with pytest.raises(ValueError):
        extended_ccm.embed_with_lag(x, 2, 0)

def test_bidirectional_causality():
    """Test bidirectional causality detection."""
    np.random.seed(42)
    x, y = generate_bidirectional_data(ts_length=3000, tau_d=2)
    
    extended_ccm = ExtendedCCM(x, y, max_dim=3)
    lags = np.arange(-8, 9)
    
    x_stable = x[100:2000]  # Use stable portion
    y_stable = y[100:2000]
    
    causality_type, ccm_skill = extended_ccm.detect_causality(
        x_stable, y_stable,
        lags,
        embed_dim=2,
        library_length=500
    )
    
    plot_ccm_skills(ccm_skill, "Bidirectional Causality: CCM Skills", 
                   "test_bidirectional", "test_results")
    
    # Print debug information
    print("\nBidirectional Test Results:")
    for lag, skill_x, skill_y in ccm_skill:
        print(f"Lag {lag:5.1f}: X→Y = {skill_x:.3f}, Y→X = {skill_y:.3f}")
    
    assert causality_type == "Bidirectional", f"Expected 'Bidirectional', got {causality_type}"

def test_synchrony():
    """Test synchrony detection."""
    np.random.seed(42)
    x, y = generate_synchrony_data(ts_length=3000)
    
    extended_ccm = ExtendedCCM(x, y, max_dim=2)
    lags = np.arange(-8, 9)
    
    x_stable = x[100:2000]
    y_stable = y[100:2000]
    
    causality_type, ccm_skill = extended_ccm.detect_causality(
        x_stable, y_stable,
        lags,
        embed_dim=2,
        library_length=200
    )
    
    plot_ccm_skills(ccm_skill, "Synchrony: CCM Skills", 
                   "test_synchrony", "test_results")
    
    # Print debug information
    print("\nSynchrony Test Results:")
    for lag, skill_x, skill_y in ccm_skill:
        print(f"Lag {lag:5.1f}: X→Y = {skill_x:.3f}, Y→X = {skill_y:.3f}")
    
    assert causality_type == "Synchrony", f"Expected 'Synchrony', got {causality_type}"

def test_transitive_causality():
    """Test transitive causality detection."""
    np.random.seed(42)
    y1, y2, y3, y4 = generate_transitive_data(ts_length=3000)
    
    extended_ccm = ExtendedCCM(y1, y4, max_dim=4)
    lags = np.arange(-8, 9)
    
    y1_stable = y1[200:2200]  # Longer stabilization period
    y4_stable = y4[200:2200]
    
    causality_type, ccm_skill = extended_ccm.detect_causality(
        y1_stable, y4_stable,
        lags,
        embed_dim=4,
        library_length=400  # Larger library for transitive case
    )
    
    plot_ccm_skills(ccm_skill, "Transitive Causality: CCM Skills", 
                   "test_transitive", "test_results")
    
    # Print debug information
    print("\nTransitive Test Results:")
    for lag, skill_x, skill_y in ccm_skill:
        print(f"Lag {lag:5.1f}: Y1→Y4 = {skill_x:.3f}, Y4→Y1 = {skill_y:.3f}")
    
    assert causality_type == "Transitive", f"Expected 'Transitive', got {causality_type}"

def test_extended_ccm_integration():
    """Test the integration of ExtendedCCM with the existing codebase."""
    test_extended_ccm()

if __name__ == "__main__":
    pytest.main([__file__])
