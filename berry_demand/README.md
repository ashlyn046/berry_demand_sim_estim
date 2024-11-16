# Berry Demand Estimation Project

## Overview
This project simulates product demand and prices over several time periods, estimates market shares, and calculates demand parameters using synthetic data. The estimation process employs Instrumental Variable (IV) and Ordinary Least Squares (OLS) regression, leveraging multiprocessing for efficient computation.

## Features
- **Data Simulation**: Generates synthetic data for demand estimation, including product characteristics, prices, and market shares.
- **Demand Sampling Methods**: Provides three approaches for drawing demand:
  - Custom discrete distribution
  - Cumulative sum distribution
  - Uniform distribution
- **Demand Estimation**: Estimates price sensitivity and product characteristic effects using IV regression.
- **Efficiency**: Utilizes Python's multiprocessing library to parallelize computations across demand sampling methods.

## Installation
1. Clone this repository.
2. Install the required Python libraries:
    ```bash
    pip install numpy pandas scipy linearmodels
    ```

## Usage
To run the project:

1. **Configure Parameters**: Open the `main.py` script and adjust the parameters as needed:
    - `N`: Number of consumers (default: 10,000)
    - `J`: Number of products, including the outside good (default: 3)
    - `T`: Number of time periods (default: 365)
    - Simulation parameters such as `beta0`, `beta1`, `alpha`, `omega0`, `omega1`, and `rho`.

2. **Execute the Script**:
    ```bash
    python main.py
    ```

3. **Process Overview**:
    - **Data Simulation**: Generates synthetic data based on specified parameters.
    - **Demand Drawing**: Samples demand using three different methods.
    - **Demand Estimation**: Estimates demand parameters (`beta1` and `alpha`) for each sampling method using IV regression in parallel.
    - **Output Results**: Displays the estimated parameters and total execution time.

## Outputs
The script produces:

- **Estimated Parameters**:
    - `beta1_hat`: Sensitivity to product characteristics.
    - `alpha_hat`: Price sensitivity.
- **Execution Time**: Total runtime for simulation, demand draws, and estimation.



## File Descriptions

### `data_simulation.py`
Simulates synthetic data for demand estimation:
- **Function**: `simulate_data(N, J, T, beta0, beta1, alpha, omega0, omega1, rho)`
- **Processes**:
    - Generates product characteristics (`x`) and input prices (`z`).
    - Models correlated error terms in price and utility equations.
    - Computes market shares using a logit model.

### `demand_draws.py`
Implements three demand sampling methods:
1. **Custom Discrete**: Matches market share probabilities.
    - **Function**: `draw_demand_custom(df, N, J, T)`
2. **Cumulative Sum**: Uses cumulative probabilities for sampling.
    - **Function**: `draw_demand_cumsum(df, N, J, T)`
3. **Uniform**: Generates demand draws with uniform random samples.
    - **Function**: `draw_demand_uniform(df, N, J, T)`

### `estimation.py`
Defines the estimation function using IV regression:
- **Function**: `estimModel(args)`
- **Processes**:
    - Calculates observed market shares and deltas.
    - Performs IV regression with input prices as instruments to estimate:
        - `beta1` (product characteristic sensitivity).
        - `alpha` (price sensitivity).

### `main.py`
Main script that orchestrates the entire process:
- **Processes**:
    1. Data simulation using `data_simulation.py`.
    2. Demand sampling using methods from `demand_draws.py`.
    3. Demand estimation using `estimation.py` in parallel processes.
- **Functionality**:
    - Measures total execution time.
    - Prints estimation results for each demand drawing method.

## Notes
- **Extensibility**: The scripts are modular and can be adapted for additional demand estimation methods or simulation adjustments.
- **Parallel Processing**: Three parallel processes handle demand estimation for the three sampling methods, significantly reducing runtime.
- **Robustness Testing**: Compare results across sampling methods to evaluate the robustness of demand estimation.

For more details, refer to the inline comments in each script.
