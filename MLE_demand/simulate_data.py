
import pandas as pd
import numpy as np

def simFlightDemand(J, T, delta, alpha, lambd):
    """
    Simulates flight demand data for J products over T time periods.

    Parameters:
    - J (int): Number of products (e.g., flights).
    - T (int): Number of time periods.
    - delta (float): Delta parameter in the logistic model.
    - alpha (float): Alpha parameter in the logistic model.
    - lambd (float): Arrival rate parameter for Poisson distribution.

    Returns:
    - DataFrame: Simulated flight demand data.
    """
    # Generate random prices for each product and time period
    p = np.random.normal(loc=50, size=J * T)
    data = {'p': p}
    df = pd.DataFrame(data)
    # Calculate market shares using the logistic model
    df["shares"] = np.exp(delta - alpha * df.p) / (1 + np.exp(delta - alpha * df.p))
    # Create identifiers for time periods and products
    df["t"] = np.repeat(np.arange(T), J)
    df["j"] = np.tile(np.arange(J), T)
    # Simulate demand using a Poisson distribution
    X_poisson = []  # Demand for each product and time period
    arrivals = []   # Total arrivals in each time period
    for idx in np.arange(T):
        pk = df[df.t == idx].shares.values  # Market shares in time period idx
        Q = []
        arrival = np.random.poisson(lambd)  # Total arrivals in time period idx
        for i in range(pk.size):
            Q.append(int(round(arrival * pk[i])))  # Demand for product i
        X_poisson.append(Q)
        arrivals.append(arrival)
    # Convert demand and arrivals to DataFrames
    X_poisson = np.array(X_poisson)
    arrivals = np.array(arrivals)
    dfS = pd.DataFrame(X_poisson)
    dfS["t"] = np.arange(T)
    dfL = pd.DataFrame(arrivals)
    dfL["t"] = np.arange(T)
    # Reshape demand data to have one row per product and time period
    dfS = dfS.reset_index(drop=True)
    dfTempS = dfS.drop("t", axis=1)
    dfS = dfS.melt(id_vars="t", value_vars=list(dfTempS.columns.values), value_name='Q')
    dfS.rename(columns={'variable': 'j'}, inplace=True)
    # Prepare arrival data
    dfL.rename(columns={0: 'lam'}, inplace=True)
    # Merge demand and arrival data
    dfM = pd.merge(dfS, dfL, on='t', how='inner')
    # Merge with original DataFrame
    df = df.merge(dfM, on=['t', 'j'])
    # Estimate lambda (arrival rate) as the mean of observed arrivals
    df['lam_hat'] = df['lam'].mean()
    return df
