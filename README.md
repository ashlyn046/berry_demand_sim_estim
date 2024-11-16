# Demand Estimation Project

## Overview
The Demand Estimation Project encompasses two specialized subprojects, `ml_estim` and `berry_demand`, each addressing distinct aspects of demand modeling and estimation. Together, they provide a comprehensive framework for simulating and estimating demand in different contexts, leveraging advanced statistical techniques and efficient computational methods.

## Subprojects

### ml_estim
**Flight Demand Estimation using Maximum Likelihood Estimation (MLE)**
- **Purpose**: Simulates flight demand and estimates parameters for a logistic demand model.
- **Key Features**:
  - Generates synthetic data using Poisson distributions.
  - Computes partial derivatives of the likelihood function.
  - Optimizes parameters (`alpha` and `delta`) with SciPy's `minimize`.
  - Visualizes objective and log-likelihood surfaces through 3D plots.
- **Read More**: [ml_estim/README.md](ml_estim/README.md)

### berry_demand
**Product Demand Estimation using IV and OLS Regression**
- **Purpose**: Simulates product demand and prices, estimates market shares, and calculates demand parameters.
- **Key Features**:
  - Generates synthetic data for demand estimation, including product characteristics and prices.
  - Implements three demand sampling methods: Custom Discrete, Cumulative Sum, and Uniform distributions.
  - Estimates price sensitivity and product characteristic effects using IV and OLS regression.
  - Utilizes multiprocessing for efficient parallel computations.
- **Read More**: [berry_demand/README.md](berry_demand/README.md)

## Project Integration
Both subprojects are integral components of the Demand Estimation Project, offering complementary approaches to understanding and modeling demand:
- **ml_estim** focuses on flight-specific demand scenarios using MLE, providing insights into parameter estimation in a controlled simulation environment.
- **berry_demand** extends the demand estimation framework to general product markets, incorporating multiple sampling methods and regression techniques to enhance robustness and applicability.

By housing these subprojects together, the Demand Estimation Project facilitates a versatile and scalable platform for demand analysis across various industries and use cases.

## Notes
- **Modularity**: Each subproject is designed to operate independently, allowing users to focus on specific demand estimation needs without unnecessary complexity.
- **Extensibility**: The modular structure supports easy integration of additional estimation methods or simulation techniques as needed.
- **Synthetic Data**: Utilizing synthetic data ensures reproducibility and controlled experimentation, essential for validating estimation methodologies.

For detailed information on each subproject, refer to their respective README files in the `ml_estim` and `berry_demand` directories.
