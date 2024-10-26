# Extended CCM: Building on skCCM

This project extends the functionality of the [skCCM](https://github.com/nickc1/skccm) library by adding additional features for handling time lags, embedding dimensions, and applying the methodology based on the extended Convergent Cross Mapping (CCM) as described by Sugihara et al.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Examples](#examples)
- [Testing](#testing)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/SharathSPhD/eskCCM.git
    cd extended-ccm
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install the `skCCM` library:
    ```bash
    pip install skccm
    ```

## Usage

### Running the examples
You can run the provided examples for both the standard CCM workflow (`eskCCM`) and the extended version based on Sugihara's work.

1. **Run the eskCCM Example**:
    ```bash
    python examples/skccm_examples.py
    ```

2. **Run the ExtendedCCM Test**:
    ```bash
    python examples/diagnostic.py
    ```

## Modules

This project includes the following modules:
- `extended_ccm.py`: Provides an extended CCM class for handling time lags and embedding dimensions.

## Testing

Unit tests are available for each module. To run the tests, use:
```bash
pytest tests/
```

## ExtendedCCM Class

The `ExtendedCCM` class extends the functionality of the standard `CCM` class by adding additional features for handling time lags, embedding dimensions, and applying the methodology based on the extended Convergent Cross Mapping (CCM) as described by Sugihara et al.

### Usage

To use the `ExtendedCCM` class, you can follow the example provided in `examples/diagnostic.py`. Here is a brief overview:

1. **Import the class**:
    ```python
    from eskCCM.extended_ccm import ExtendedCCM
    ```

2. **Generate data**:
    ```python
    x, y = generate_bidirectional_data()
    ```

3. **Initialize the ExtendedCCM object**:
    ```python
    extended_ccm = ExtendedCCM(x, y, max_dim=3)
    ```

4. **Detect causality**:
    ```python
    causality_type, ccm_skill = extended_ccm.detect_causality(
        x_stable, y_stable,
        lags,
        embed_dim=2,
        library_length=500
    )
    ```

5. **Plot results**:
    ```python
    plot_ccm_skills(ccm_skill, "ExtendedCCM: Bidirectional Causality", 
                   "extended_ccm_bidirectional", "results")
    ```

For more details, refer to the `examples/diagnostic.py` file.
