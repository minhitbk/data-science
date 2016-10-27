
# coding: utf-8

# In[ ]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


# In[ ]:

# Read data from csv file to pandas dataframe
df = pd.read_csv("data/training.csv", header=0)

# Transform data into a format that is easier for analysis
corr_df = pd.DataFrame({"Series1": df[df.serieNames=="serie_1"].sales
                            .reset_index().drop(["index"], axis=1).sales,
                        "Series2": df[df.serieNames=="serie_2"].sales
                            .reset_index().drop(["index"], axis=1).sales})


# In[ ]:

# Plot Series1 & Series2
fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 2)

fig.add_subplot(gs[0, 0:1])
plt.title("Series1")
plt.xlabel("Date")
plt.ylabel("Sales")
corr_df["Series1"].plot()

fig.add_subplot(gs[0, 1:2])
plt.title("Series2")
plt.xlabel("Date")
plt.ylabel("Sales")
corr_df["Series2"].plot()

# Plot histogram of Series1 and Series2
fig = plt.figure(figsize=(16, 6))

fig.add_subplot(gs[0, 0:1])
plt.title("Series1")
plt.xlabel("Sales")
corr_df["Series1"].plot(kind='hist', bins=100)

fig.add_subplot(gs[0, 1:2])
plt.title("Series2")
plt.xlabel("Sales")
corr_df["Series2"].plot(kind='hist', bins=100)

# ACF function
def acorr(x, ax, maxlags):
    x = x - x.mean()
    autocorr = np.correlate(x, x, mode='full')
    autocorr /= autocorr.max()
    autocorr = autocorr[x.size: x.size + maxlags]
    
    return ax.plot(autocorr)

# Plot ACF of Series1 and Series2
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
axes[0].set_title("Series1")
axes[0].set_xlabel("Lags")
axes[0].set_ylabel("ACF")
axes[0].set_ylim([-0.1, 1])
acorr(np.float32(corr_df["Series1"]), axes[0], 200)

axes[1].set_title("Series2")
axes[1].set_xlabel("Lags")
axes[1].set_ylabel("ACF")
axes[1].set_ylim([-0.1, 1])
acorr(np.float32(corr_df["Series2"]), axes[1], 200)


# In[ ]:

# Summary statistics of the 2 products
df["sales"].groupby(df["serieNames"]).describe()


# In[ ]:

# Cross-Correlation between the 2 products
corr_df.corr()

