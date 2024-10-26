import skCCM as ccm
import skCCM.data as data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skCCM.utilities import train_test_split

# Configure seaborn for better visuals
sns.set_style('ticks')
sns.set_context('paper', font_scale=1.5)

# 1. Generate coupled logistic map data
rx1 = 3.72
rx2 = 3.72
b12 = 0.2
b21 = 0.01
ts_length = 1000
x1, x2 = data.coupled_logistic(rx1, rx2, b12, b21, ts_length)

# Plot the time series
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
ax[0].plot(x1[:100])
ax[1].plot(x2[:100])
ax[0].set_yticks([.1, .3, .5, .7, .9])
ax[1].set_xlabel('Time')
sns.despine()
plt.show()

# 2. Calculate mutual information
e1 = ccm.Embed(x1)
e2 = ccm.Embed(x2)
mi1 = e1.mutual_information(10)
mi2 = e2.mutual_information(10)

# Plot mutual information
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
ax[0].plot(np.arange(1, 11), mi1)
ax[1].plot(np.arange(1, 11), mi2)
ax[1].set_xlabel('Lag')
sns.despine()
plt.show()

# 3. Embed the time series
lag = 1
embed = 2
X1 = e1.embed_vectors_1d(lag, embed)
X2 = e2.embed_vectors_1d(lag, embed)

# Split into training and testing sets
x1tr, x1te, x2tr, x2te = train_test_split(X1, X2, percent=.75)

# 4. Initialize CCM model and fit
ccm_model = ccm.CCM()
ccm_model.fit(x1tr, x2tr)

# Test for a range of library lengths
lib_lens = np.arange(10, len(x1tr), len(x1tr)//20, dtype='int')
x1_pred, x2_pred = ccm_model.predict(x1te, x2te, lib_lengths=lib_lens)

# Score the predictions
score_x1, score_x2 = ccm_model.score()
print(f"Score X1: {score_x1}")
print(f"Score X2: {score_x2}")

# Plot the forecast skill across different library lengths
plt.plot(lib_lens, score_x1, label='X1 Prediction Skill')
plt.plot(lib_lens, score_x2, label='X2 Prediction Skill')
plt.xlabel('Library Length')
plt.ylabel('Prediction Skill')
plt.legend()
sns.despine()
plt.show()
