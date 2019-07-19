"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., Nearing, G., "Prediction 
in Ungauged Basins with Long Short-Term Memory Networks". submitted to Water Resources Research 
(2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import pickle
import sys

import numpy as np
import pandas as pd

from performance_functions import (baseflow_index, bias, flow_duration_curve, get_quant, high_flows,
                                   low_flows, nse, stdev_rat, zero_freq)

# file name of ensemble dictionary is a user input
experiment = sys.argv[1]

# load ensemble file
fname = f"results_data/{experiment}.pkl"
with open(fname, 'rb') as f:
    ens_dict = pickle.load(f)

# calcualte performance measures for ensembles
stats = []
bdex = -1
for basin in ens_dict:

    # list columns in dataframe
    sim_cols = ens_dict[basin].filter(regex="qsim")
    _, nMembers = ens_dict[basin].filter(regex="qsim").shape

    # calcualte ensemble mean performance metrics
    obs5, sim5 = get_quant(ens_dict[basin], 0.05)
    obs95, sim95 = get_quant(ens_dict[basin], 0.95)
    obs0, sim0 = zero_freq(ens_dict[basin])
    obsH, simH = high_flows(ens_dict[basin])
    obsL, simL = low_flows(ens_dict[basin])
    e_nse = nse(ens_dict[basin])
    e_bias = bias(ens_dict[basin])
    e_stdev_rat = stdev_rat(ens_dict[basin])
    #  obsBF, simBF = baseflow_index(ens_dict[basin])
    obsFDC, simFDC = flow_duration_curve(ens_dict[basin])

    # add ensemble mean stats to globaldictionary
    stats.append({
        'basin': basin,
        'nse': e_nse,
        'bias': e_bias,
        'stdev': e_stdev_rat,
        'obs5': obs5,
        'sim5': sim5,
        'obs95': obs95,
        'sim95': sim95,
        'obs0': obs0,
        'sim0': sim0,
        'obsL': obsL,
        'simL': simL,
        'obsH': obsH,
        'simH': simH,
        'obsFDC': obsFDC,
        'simFDC': simFDC
    })

    # print basin-specific stats
    bdex = bdex + 1
    print(f"{basin} ({bdex} of {len(ens_dict)}) --- NSE: {stats[bdex]['nse']}")

# save ensemble stats as a csv file
stats = pd.DataFrame(stats,
                     columns=[
                         'basin', 'nse', 'bias', 'stdev', 'obs5', 'sim5', 'obs95', 'sim95', 'obs0',
                         'sim0', 'obsL', 'simL', 'obsH', 'simH', 'obsFDC', 'simFDC'
                     ])
fname = f"stats/{experiment}.csv"
stats.to_csv(fname)

# print to screen
print('Mean NSE: ', stats['nse'].mean())
print('Median NSE: ', stats['nse'].median())
print('Num Failures: ', np.sum((stats['nse'] < 0).values.ravel()))
