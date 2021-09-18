#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().system('pip install pandas_datareader')
from pandas_datareader import data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matploitlib', 'inline')


# In[8]:


test = data.DataReader(['TSLA', 'FB'], 'yahoo', start='2016/01/01', end='2021/04/30')
test.head(20)


# In[10]:


test = test['Adj Close']
test.head(20)


# In[11]:


tesla = test['TSLA'].pct_change().apply(lambda x: np.log(1+x))
tesla.head()


# In[12]:


var_tesla = tesla.var()
var_tesla


# In[13]:


facebook = test['FB'].pct_change().apply(lambda x: np.log(1+x))
facebook.head()


# In[14]:


var_facebook = facebook.var()
var_facebook


# In[15]:


tesla_volatility = np.sqrt(var_tesla * 250)
facebook_volatility = np.sqrt(var_facebook *250)
tesla_volatility , facebook_volatility


# In[16]:


test.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250)).plot(kind='bar')


# In[17]:


test1 = test.pct_change().apply(lambda x: np.log(1+x))
test1.head()


# In[18]:


test1['TSLA'].cov(test1['FB'])


# In[19]:


test1['TSLA'].corr(test1['FB'])


# In[20]:


test2 = test.pct_change().apply(lambda x: np.log(1+x))
test2.head()


# In[21]:


w = [0.2, 0.8]
e_r_ind = test2.mean()
e_r_ind


# In[22]:


e_r = (e_r_ind*w).sum()
e_r


# In[2]:


df = data.DataReader(['AAPL', 'NKE', 'GOOGL', 'AMZN'], 'yahoo', start='2016/01/01', end='2021/04/30')
df.head()


# In[3]:


df = df['Adj Close'] #Close pricing of the company's stock on the given day
df.head()


# # Covariance
#  
# Covariance measures the directional relationship between the returns on two assets.
# 
# A positive covariance means that returns of the two assets move together while a negative covariance means they move inversely. Risk and volatility can be reduced in a portfolio by pairing assets that have a negative covariance.
# 
# ## cov_matrix
# 
# Representing the Covariance of all the stocks using matrix for better understanding.
# 
# ## Why are we taking log?
# 
# This is beacuse the log of the returns is <b>time additive</b>.
# 
# That is,
# If r13 is the returns for time between t3 and t1.
# r12 is the returns between t1 and t2 and
# r23 is the returns between t2 and t3
# 
# Then, log(r13) = log(r12) + log(r23)
# 
# For example:,
# If p1 = 100, p2 = 110 and p3 = 120,
# where p1 is price of stock in time 1
# 
# Then:
# 
# log(r12) = ln(p2/p1) = ln(110/100) = 9.53%
# log(r23) = ln(120/110) = 8.7% and
# log(r13) = log(r12) + log(r23) = 9.53 + 8.7 = 18.23%, which is same as ln(120/100).
# 
# This means a log change of +0.1 today and then -0.1 tomorrow will give you the same value of stock as yesterday. This is not true if you simply compute percentage chang

# In[4]:


cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix


# # Correlation
# 
# Correlation, in the finance and investment industries, is a statistic that measures the degree to which two securities move in relation to each other. Correlations are used in advanced portfolio management, computed as the correlation coefficient, which has a value that must fall between -1.0 and +1.0.
# 
# We can think of correlation as a scaled version of covariance, where the values are restricted to lie between -1 and +1.
# 
# A correlation of -1 means negative relation, i.e, if correlation between Asset A and Asset B is -1, if Asset A increases, Asset B decreases.
# 
# A correlation of +1 means positive relation, i.e, if correlation between Asset A and Asset B is 1, if Asset A increases, Asset B increases.
# 
# A correlation of 0 means no relation, i.e, if correlation between Asset A and Asset B is 0, they dont have any effect on each other.
# 
# ## corr_matrix
# 
# Correlation matrix

# In[5]:


corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
corr_matrix


# # Weights
# 
# Let’s define an array of random weights for the purpose of calculation. These weights will represent the percentage allocation of investments between these two stocks. They must add up to 1.
# 
# So, the problem of portfolio optimization is nothing but to find the optimal values of weights that maximizes expected returns while minimizing the risk (standard deviation).

# In[6]:


w = {'AAPL': 0.1, 'NKE': 0.2, 'GOOGL': 0.5, 'AMZN': 0.2}
port_var = cov_matrix.mul(w, axis=0).mul(w, axis=1).sum().sum()
port_var


# # Portfolio Expected Returns
# 
# The mean of returns (given by change in prices of asset stock prices) give us the expected returns of that asset.
# The sum of all individual expected returns further multiplied by the weight of assets give us expected return for the portfolio.
# 
# We use the <b>resample()</b> function to get yearly returns. The argument to function, ‘Y’, denotes yearly.
# If we dont perform resampling, we will get daily returns.

# In[7]:


ind_er = df.resample('Y').last().pct_change().mean()
ind_er


# In[8]:


w = [0.1, 0.2, 0.5, 0.2]
port_er = (w*ind_er).sum()
port_er


# # Plotting the Efficient Frontier
# 
# Efficient frontier is a graph with ‘returns’ on the Y-axis and ‘volatility’ on the X-axis. It shows us the maximum return we can get for a set level of volatility, or conversely, the volatility that we need to accept for certain level of returns.

# In[9]:


#Volatility is given by annual standard deviation. It is multiplied by 250 because there are 250 trading days in a year.
ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
ann_sd


# In[10]:


assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
assets.columns = ['Returns', 'Volatility']
assets


# # Analysis of the above Results
# 
# Apple has the highest Volatility or we can say Risk, but also the highest return.<br>
# Google has the lowest volatility but, the lowest porfit or return as well.
# 
# Next, to plot the graph of efficient frontier, we need run a loop. In each iteration, the loop considers different weights for assets and calculates the return and volatility of that particular portfolio combination.
# 
# We run this loop a 1000 times.
# 
# To get random numbers for weights, we use the "np.random.random()" function. But, the sum of weights must be 1, so we divide those weights by their cumulative sum.

# In[11]:


p_ret = [] # Define an empty array for portfolio returns
p_vol = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights

num_assets = len(df.columns)
num_portfolios = 10000


# In[12]:


for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
    p_vol.append(ann_sd)


# In[13]:


data = {'Returns':p_ret, 'Volatility':p_vol}

for counter, symbol in enumerate(df.columns.tolist()):
    #print(counter, symbol)
    data[symbol+' weight'] = [w[counter] for w in p_weights]


# In[14]:


portfolios  = pd.DataFrame(data)
portfolios.head() 


# You can see that there are a number of portfolios with different weights, returns and volatility. Plotting the returns and volatility from this dataframe will give us the efficient frontier for our portfolio.

# In[15]:


portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])


# # Reading data form the Efficient Frontier
# 
# Each point on the line (left edge) represents an optimal portfolio of stocks that maximises the returns for any given level of risk.
# 
# 
# The point (portfolios) in the interior are sub-optimal for a given risk level. For every interior point, there is another that offers higher returns for the same risk.
# 
# On this graph, we can also see the combination of weights that will give you all possible combinations:
# 
# <ul>
#     <li>Minimum volatility (left most point)</li>
#     <li>Maximum returns (top most point)</li>
# </ul>
# 
# And everything in between.

# In[16]:


min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
# idxmin() gives us the minimum value in the column specified.                               
min_vol_port


# The minimum volatility is in a portfolio where the weights of Apple, Nike, Google and Amazon are 11%, 38%, 31% and 19% respectively. This point can be plotted on the efficient frontier graph as shown:

# In[17]:


plt.subplots(figsize=[10,10])
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)


# The red star denotes the most efficient portfolio with minimum volatility.
# 
# It is worthwhile to note that any point to the right of efficient frontier boundary is a sup-optimal portfolio.
# 
# We found the portfolio with minimum volatility, but you will notice that the return on this portfolio is pretty low. Any sensible investor wants to maximize his return, even if it is a tradeoff with some level of risk.
# 
# The question arises that how do we find this optimal risky portfolio and finally optimize our portfolio to the maximum?
# 
# This is done by using a parameter called the <b>Sharpe Ratio</b>.
# 
# The ratio is the average return earned in excess of the risk-free rate per unit of volatility or total risk. Volatility is a measure of the price fluctuations of an asset or portfolio.
# 
# 
# The risk-free rate of return is the return on an investment with zero risk, meaning it’s the return investors could expect for taking no risk.
# 
# The optimal risky portfolio is the one with the highest Sharpe ratio.

# In[18]:


rf = 0.01 # risk factor
optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
optimal_risky_port


# You can notice that while the difference in risk between minimum volatility portfolio and optimal risky portfolio is just 4%, the difference in returns is a whopping 12%.
# We can plot this point too on the graph of efficient frontier.

# In[19]:


plt.subplots(figsize=(10, 10))
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)


# The green star represents the <i>Optimal Risky Portfolio</i>.

# In[ ]:




