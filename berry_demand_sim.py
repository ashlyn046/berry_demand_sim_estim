# -----------------------------------------------------------------------------------------------------------------
# SCRIPT DESCRIPTION:
# This script models demand estimation for multiple firms over several time periods using both IV (Instrumental 
# Variable) and OLS (Ordinary Least Squares) regression. It generates synthetic data with correlated errors to simulate 
# product demand and prices. The estimation function, `estimModel`, calculates market shares and uses IV to instrument 
# prices with input prices. Demand draws are sampled from three distributions (custom discrete, cumulative sum, 
# and uniform), and the multiprocessing library is used to parallelize the estimation process for faster computation.
# The results are printed along with the total execution time for performance tracking.
# -----------------------------------------------------------------------------------------------------------------

from multiprocessing import Pool
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from linearmodels.iv.model import IV2SLS
import statsmodels.api as sm
from scipy import stats
import time

# DEFINING FUNCTION FOR ESTIMATION WITH DIFFERENT DEMAND DRAWS
def estimModel(args):
	X_curr, df, N, J, T = args
	#
	# Taking in the demand draw (X_curr), and putting it into a df with time period
	X_curr = np.array(X_curr)
	dfS = pd.DataFrame(X_curr) #only 365 time periods
	dfS["t"] = np.arange(T)
	#
	# Reshaping so that demand (Q) is all in one column
	dfS = dfS.reset_index()
	dfTemp = dfS.drop("index", axis=1)
	dfTemp = dfTemp.drop("t", axis=1)
	dfS = dfS.melt(id_vars="t",value_vars=list(dfTemp.columns.values), value_name = 'Q')
	dfS.rename(columns={'variable': 'j'}, inplace=True)
	#
	# Merging back into the original data
	df = df.merge(dfS, on=['t', 'j'])
	#
	# Generating market shares based on demand and number of consumers
	df["s_data"] = df.Q / N
	#
	# Generating market shares for the outside good
	dfs0 = df.loc[df.j == 0] # getting obs where j=0 (outside good)
	dfs0 = dfs0[["t", "s_data"]] # making a df with just time and market shares for j=0
	dfs0.rename(columns = {"s_data" : "s0"}, inplace = True)
	df = df.merge(dfs0, on = "t") # merging back in to the original data
	df.rename(columns = {"s0_x" : "s0"}, inplace=True)
	#
	# Calculating delta based on market shares
	df["delta2"] = np.log(df.s_data) - np.log(df.s0)
	#
	# Making a new df with only obs that are NOT the outside option
	df1 = df.loc[df.j != 0]
	#
	# Running IV, instrumenting input prices for prices
	iv_model = IV2SLS(dependent=df1['delta2'], exog=df1['x'], endog=df1['p'], instruments=df1['input_prices'])
	iv_results = iv_model.fit()
	#OLS model to comp
	X = sm.add_constant(df1['x'])
	ols_model = sm.OLS(df1['delta2'], X)
	ols_results = ols_model.fit().summary()
	#
	# Returning
	return iv_results.params[0], iv_results.params[1]

# MAIN FUNCTION
if __name__ == '__main__':
	t0 = time.time()
	#
	# Picking N: num consumers, J: num firms/goods, T: num time periods
	N = 10000
	J = 3 # (4 inside goods and 1 outside good)
	T = 365 
	#
	# u_ijt = beta0 + beta1*x_jt - alpha*p_jt + xi_jt + ep_ijt
	#
	# Picking parameters to generate data
	beta0 = 0
	beta1 = .5
	alpha = .18
	#
	# Generating x from a random normal dist, and z to help make p
	x = np.random.normal(size = J * T)
	z = np.random.normal(size = J * T)
	#
	# p = omega0 + omega1*z + nu
	omega0 = 1
	omega1 = .5
	# other error term is xi
	rho = .9
	mean = [0, 0]
	cov = [[1, rho], [rho, 1]] 
	error_terms = multivariate_normal.rvs(mean, cov, size= J * T) # generating correlated errors
	nu = error_terms[:,0]/2
	xi = error_terms[:,1]/15
	#
	p = omega0 - omega1*z + nu #now p is correlated with xi
	#
	# Generate input prices by adding noise to prices (without nu) (I do this because I want input prices and prices to be correlated)
	input_prices = omega0 - omega1*z
	#
	data = {
	    'x' : x,
	    'xi': xi, 
	    'p': p,
	    'input_prices': input_prices
	}
	#
	df = pd.DataFrame(data)
	#
	#CHECKING CORRELATIONS
	# correlation = df['input_prices'].corr(df['p'])
	# print(correlation)
	# correlation = df['xi'].corr(df['p'])
	# print(correlation)
	# correlation = df['xi'].corr(df['input_prices'])
	# print(correlation)
	#
	#Calculating what delta would be based on our chosen parameters
	df["delta1"] = beta0 + beta1*df.x - alpha*df.p + df.xi
	#
	#Making rows for j and t
	df["t"] = np.repeat(np.arange(T), J)
	df["j"] = np.tile(np.arange(J), T)
	#
	#Setting delta1 to 0 wherever j=0 (outside option)
	df.loc[df.j == 0, "delta1"] = 0
	df["edelta"] = np.exp(df.delta1)
	df["sj"] = df.edelta / df.groupby("t").edelta.transform("sum") # product shares (add up to 1 in each period) (logit)
	#
	# DRAWING DEMAND from 3 different distributions
	# could loop through custom dataframe 
	#
	# Custom discrete distribution
	X_custm = []
	for idx in np.arange(T):
		pk = df.loc[df.t == idx].sj.values #sj in given time period
		xk = np.arange(J)
		custm = stats.rv_discrete(name='custm', values=(xk, pk)) #create a custom discrete probability distribution where xk are possible outcomes and pk are probabilities
		Q = custm.rvs(size=N) #take N random samples from discrete distribution
		Q = np.unique(Q, return_counts=True)[1] # count how many times each val appears in Q and stores frequencies in Q
		X_custm.append(Q)

	# Cumulative sum distribution
	X_cumsum = []
	for idx in np.arange(T):
		pk = df.loc[df.t == idx].sj.values #sj in given time period
		xk = np.arange(J)
		Q = np.searchsorted(np.cumsum(pk), np.random.random(N)) # sample from cumulative sum distribution
		Q = np.unique(Q, return_counts=True)[1] # count how many times each val appears in Q and stores frequencies in Q
		X_cumsum.append(Q)

	# Uniform Distirbution
	X_unif = []
	for idx in np.arange(T):
		pk = df.loc[df.t == idx].sj.values #sj in given time period
		xk = np.arange(J)
		Q = np.searchsorted(np.cumsum(pk), np.random.uniform(size=N))
		Q = np.bincount(Q, minlength=J)  # Count the frequencies of each value in Q
		X_unif.append(Q)
	#
	# Running estimation function for all at once
	with Pool(5) as p:
		results = p.map(estimModel, [(X_custm, df, N, J, T), (X_cumsum, df, N, J, T), (X_unif, df, N, J, T)])
	#
	p.close()
	p.join()
	print(results)
	#
	t1 = time.time()
	total = t1-t0
	print("Time: ", total)


# time on cumsum: 0.8805520534515381
# time on custm: 0.5985548496246338
# pk = df.loc[df.t == 0].sj.values #sj in given time period (We have J values)
# print("pk: ", pk)
# xk = np.arange(J)
# print("xk: ", xk) # there are J values here

# Q = np.searchsorted(np.random.uniform(pk), np.random.random(N)) # sample from cumulative sum distribution
# print(Q)

# Q = np.bincount(Q, minlength=J)[1:J+1]
# #Q = np.unique(Q, return_counts=True)[1] # count how many times each val appears in Q and stores frequencies in Q
# print(Q)
# X_unif.append(Q)
# print(X_unif)

# for idx in np.arange(T):
# 	pk = df.loc[df.t == idx].sj.values #sj in given time period
# 	xk = np.arange(J)
# 	Q = np.searchsorted(np.random.uniform(pk), np.random.random(N)) # sample from cumulative sum distribution
# 	print(Q)
# 	Q = np.bincount(Q, minlength=J)[1:J+1] # count how many times each val appears in Q and stores frequencies in Q
# 	print(Q)
# 	X_unif.append(Q)




# ols_model = AbsorbingLS(dependent=df1['delta2'], exog=df1[['x', 'p']], absorb=None)
# ols_results = ols_model.fit()
	 
#print(iv_results.summary)
#print(iv_results.params)
#print(type(iv_results.params))

#printing beta


# for idx in np.arange(T):
#     xk = np.arange(J)
#     #Q = np.random.choice(xk, size=N)  # Sample from a uniform distribution
#     Q = np.searchsorted(np.cumsum(pk), np.random.uniform(size=N))
#     print(Q)
#     Q = np.bincount(Q, minlength=J)  # Count the frequencies of each value in Q
#     print(Q)
#     X_unif.append(Q)





