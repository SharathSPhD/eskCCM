
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
    git clone https://github.com/your-repo/extended-ccm.git
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
You can run the provided examples for both the standard CCM workflow (`skCCM`) and the extended version based on Sugihara's work.

1. **Run the skCCM Example**:
    ```bash
    python examples/skccm_examples.py
    ```

2. **Run the Sugihara Extended Example**:
    ```bash
    python examples/sugihara_extended_examples.py
    ```

## Modules

This project includes the following modules:
- `extended_ccm.py`: Provides an extended CCM class for handling time lags and embedding dimensions.
- `mutual_information.py`: Functions for calculating mutual information to determine optimal lag.
- `false_nearest_neighbors.py`: Functions for calculating the optimal embedding dimension using the False Nearest Neighbors (FNN) method.

## Testing

Unit tests are available for each module. To run the tests, use:
```bash
pytest tests/
```
