"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., Nearing, G., "Prediction 
in Ungauged Basins with Long Short-Term Memory Networks". submitted to Water Resources Research 
(2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import numpy as np


def nse(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    SSerr = np.mean(np.square(df["qsim"][idex] - df["qobs"][idex]))
    SStot = np.mean(np.square(df["qobs"][idex] - np.mean(df["qobs"][idex])))
    return 1 - SSerr / SStot


def get_quant(df, quant):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    sim = df["qsim"][idex].quantile(quant)
    obs = df["qobs"][idex].quantile(quant)
    return obs, sim


def bias(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    sim = df["qsim"][idex].mean()
    obs = df["qobs"][idex].mean()
    return (obs - sim) / obs


def stdev_rat(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    sim = df["qsim"][idex].std()
    obs = df["qobs"][idex].std()
    return sim / obs


def zero_freq(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    sim = (df["qsim"][idex] == 0).astype(int).sum()
    obs = (df["qobs"][idex] == 0).astype(int).sum()
    return obs, sim


def flow_duration_curve(df):
    obs33, sim33 = get_quant(df, 0.33)
    obs66, sim66 = get_quant(df, 0.66)
    sim = (np.log(sim33) - np.log(sim66)) / (0.66 - 0.33)
    obs = (np.log(obs33) - np.log(obs66)) / (0.66 - 0.33)
    return obs, sim


def baseflow_index(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    obsQ = df["qobs"][idex].values
    simQ = df["qsim"][idex].values
    nTimes = len(obsQ)

    obsQD = np.full(nTimes, np.nan)
    simQD = np.full(nTimes, np.nan)
    obsQD[0] = obsQ[0]
    simQD[0] = simQ[0]

    c = 0.925
    for t in range(1, nTimes):
        obsQD[t] = c * obsQD[t - 1] + (1 + c) / 2 * (obsQ[t] - obsQ[t - 1])
        simQD[t] = c * simQD[t - 1] + (1 + c) / 2 * (simQ[t] - simQ[t - 1])

    obsQB = obsQ - obsQD
    simQB = simQ - simQD

    obs = np.mean(np.divide(obsQB[1:], obsQ[1:]))
    sim = np.mean(np.divide(simQB[1:], simQ[1:]))
    return obs, sim


def high_flows(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    obsMedian = df["qobs"][idex].median()
    obsFreq = len(df["qobs"][idex].index[(df["qobs"][idex] >= 9 * obsMedian)].tolist())
    simMedian = df["qsim"][idex].median()
    simFreq = len(df["qsim"][idex].index[(df["qsim"][idex] >= 9 * simMedian)].tolist())
    return obsFreq, simFreq


def low_flows(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    obsMedian = df["qobs"][idex].median()
    obsFreq = len(df["qobs"][idex].index[(df["qobs"][idex] <= 0.2 * obsMedian)].tolist())
    simMedian = df["qsim"][idex].median()
    simFreq = len(df["qsim"][idex].index[(df["qsim"][idex] <= 0.2 * simMedian)].tolist())
    return obsFreq, simFreq
