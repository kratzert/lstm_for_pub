"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., Nearing, G., "Prediction 
in Ungauged Basins with Long Short-Term Memory Networks". submitted to Water Resources Research 
(2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""


import glob
import os
import pickle
import sys

import pandas as pd

# number of ensemble members
nSplits = 12
nSeeds = 10
firstSeed = 200

# user inputs
experiment = sys.argv[1]
gpu = sys.argv[2]

# This loop will run the evaluation procedure for all splits of all PUB ensembles
for seed in range(firstSeed, firstSeed + nSeeds):  # loop through randomized ensemble
    #for seed in range(203,209):
    for split in range(nSplits):  # number of k-fold splits

        # get the correct run directory by reading the screen report
        fname = f"reports/{experiment}.{seed}.{split}.out"
        print(f"Working on seed: {seed} -- file: {fname}")
        f = open(fname)
        lines = f.readlines()

        split_file = lines[13].split(': ')[1][:-1]
        split_num = lines[12].split(' ')[1][:-1]
        run_dir = lines[28].split('attributes in ')[1].split('attributes')[0]

        run_command = f"python3 main.py --gpu={gpu} --run_dir={run_dir} --split={split_num} --split_file={split_file} evaluate"
        #    os.system(run_command)

        # grab the test output file for this split
        file_seed = run_dir.split('seed')[1][:-1]
        results_file = glob.glob(f"{run_dir}/*lstm*seed{file_seed}.p")[0]
        with open(results_file, 'rb') as f:
            partial_dict = pickle.load(f)

        # store in a dictionary for this seed
        if split > 0:
            seed_dict.update(partial_dict)
        else:
            seed_dict = partial_dict

        # screen report
        print(seed, split, len(seed_dict))

        # --- end of split loop -----------------------------------------

    # create the ensemble dictionary
    for basin in seed_dict:
        seed_dict[basin].rename(columns={'qsim': f"qsim_{seed}"}, inplace=True)
    if seed == 200:
        ens_dict = seed_dict
    else:
        for basin in seed_dict:
            ens_dict[basin] = pd.merge(ens_dict[basin],
                                       seed_dict[basin][f"qsim_{seed}"],
                                       how='inner',
                                       left_index=True,
                                       right_index=True)

# --- end of seed loop -----------------------------------------

# calculate ensemble mean
for basin in ens_dict:
    simdf = ens_dict[basin].filter(regex='qsim_')
    ensMean = simdf.mean(axis=1)
    ens_dict[basin].insert(0, 'qsim', ensMean)

# save the ensemble results as a pickle
fname = f"analysis/results_data/{experiment}.pkl"
with open(fname, 'wb') as f:
    pickle.dump(ens_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
