
import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS
# Optionally, you can import statsmodels.api as sm if you need OLS comparisons

def estimModel(args):
    """
    Estimation function using IV regression.

    Parameters:
    - args (tuple): Contains (X_curr, df, N, J, T)

    Returns:
    - tuple: Estimated coefficients from IV regression (beta1_hat, alpha_hat)
    """
    X_curr, df, N, J, T = args

    # Create DataFrame for demand draws
    X_curr = np.array(X_curr)
    dfS = pd.DataFrame(X_curr)
    dfS["t"] = np.arange(T)

    # Reshape demand data to long format
    dfS = dfS.reset_index(drop=True)
    dfTemp = dfS.drop("t", axis=1)
    dfS = dfS.melt(id_vars="t", value_vars=list(dfTemp.columns.values), value_name='Q')
    dfS.rename(columns={'variable': 'j'}, inplace=True)
    dfS['j'] = dfS['j'].astype(int)

    # Merge demand data with original DataFrame
    df = df.merge(dfS, on=['t', 'j'])

    # Calculate market shares based on demand
    df["s_data"] = df.Q / N

    # Calculate market share of the outside good
    dfs0 = df.loc[df.j == 0, ["t", "s_data"]].rename(columns={"s_data": "s0"})
    df = df.merge(dfs0, on="t", how='left')

    # Calculate delta based on observed market shares
    df["delta2"] = np.log(df.s_data) - np.log(df.s0)

    # Filter out the outside good
    df1 = df[df.j != 0]

    # Prepare variables for IV regression
    y = df1['delta2']
    X = df1[['x']]
    endog = df1['p']
    instruments = df1[['input_prices']]

    # Run IV regression
    iv_model = IV2SLS(dependent=y, exog=X, endog=endog, instruments=instruments)
    iv_results = iv_model.fit()

    # Extract estimated coefficients
    beta1_hat = iv_results.params['x']
    alpha_hat = -iv_results.params['p']  # Negative sign because utility is U = beta1*x - alpha*p

    return beta1_hat, alpha_hat
