"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., Nearing, G., "Prediction 
in Ungauged Basins with Long Short-Term Memory Networks". submitted to Water Resources Research 
(2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import pickle

import numpy as np

FORC_VARS = ['PRCP(mm/day)', 'SRAD(W/m2)', 'Tmax(C)', 'Tmin(C)', 'Vp(Pa)']

fname = '../upstream/data/extracted/dynamic_input.pkl'
with open(fname, 'rb') as handle:
    dynamic_input_df = pickle.load(handle)

nBasins = len(dynamic_input_df)

mu = {FORC_VARS[0]: 0, FORC_VARS[1]: 0, FORC_VARS[2]: 0, FORC_VARS[3]: 0, FORC_VARS[4]: 0}

sg = {FORC_VARS[0]: 0, FORC_VARS[1]: 0, FORC_VARS[2]: 0, FORC_VARS[3]: 0, FORC_VARS[4]: 0}

for b in range(nBasins):
    for v in FORC_VARS:
        mu[v] = mu[v] + dynamic_input_df[b][v].mean() / nBasins

for b in range(nBasins):
    for v in FORC_VARS:
        sg[v] = sg[v] + ((dynamic_input_df[b][v] - mu[v])**2).mean() / nBasins

for v in FORC_VARS:
    sg[v] = np.sqrt(sg[v])

print(mu)
print(sg)
