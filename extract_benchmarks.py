"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. ( 2019). 
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065 

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import pickle

import numpy as np
import pandas as pd

SCE_SEEDS = ['05', '11', '27', '33', '48', '59', '66', '72', '80', '94']

val_start = pd.to_datetime('01101995', format='%d%m%Y')
val_end = pd.to_datetime('30092010', format='%d%m%Y')


def _read_gauge_info(path):

    huc2s = []
    gauge_ids = []
    gauge_names = []
    lats = []
    lngs = []
    drainage_areas = []

    with open(path) as gauge_info_file:

        for row in gauge_info_file:
            huc2s.append(row[:2])
            gauge_ids.append(row[3:11])
            gauge_names.append(row[11:65].strip())
            lats.append(float(row[64:74]))
            lngs.append(float(row[74:85]))
            drainage_areas.append(float(row[85:]))

    gauge_info = pd.DataFrame({
        'huc2': huc2s,
        'gauge_id': gauge_ids,
        'gauge_name': gauge_names,
        'lat': lats,
        'lng': lngs,
        'drainage_area': drainage_areas
    })
    gauge_info['gauge_str'] = gauge_info['gauge_id']
    gauge_info['gauge_id'] = gauge_info['gauge_id'].apply(pd.to_numeric)
    # gauge_info.set_index('gauge_id', inplace=True)

    return gauge_info


# load basin list
with open('analysis/results_data/global_lstm_static.pkl', 'rb') as f:
    master_dict = pickle.load(f)
basin_list = master_dict.keys()

# --- Metadata and Catchment Characteristics ---------------------------

# The purpose of loading this metadata file is to get huc and basin IDs for
# constructing model output file names.
# we also need the gauge areas for normalizing NWM output.

# load metadata file (with hucs)
meta_df = _read_gauge_info('data/basin_dataset_public_v1p2/basin_metadata/gauge_information.txt')
assert meta_df['gauge_id'].is_unique  # make sure no basins or IDs are repeated

# load characteristics file (with areas)
fname = 'data/camels_chars.txt'  # catchment characteristics file name
char_df = pd.read_table(fname, delimiter=',', dtype={'gauge_id': int})  # read characteristics file
assert char_df['gauge_id'].is_unique  # make sure no basins or IDs are repeated

# concatenate catchment characteristics with meta data
meta_df = meta_df.round({
    'lat': 5,
    'lng': 5
})  # latitudes and longitudes should be to 5 significant digits
char_df = char_df.round({'gauge_lat': 5, 'gauge_lon': 5})
assert meta_df['gauge_id'].equals(
    char_df['gauge_id'])  # check catchmenet chars & metdata have the same basins
assert meta_df['lat'].equals(char_df['gauge_lat'])  # check that latitudes and longitudes match
assert meta_df['lng'].equals(char_df['gauge_lon'])
static_df = char_df.join(
    meta_df.set_index('gauge_id'),
    on='gauge_id')  # turn into a single dataframe (only need huc from meta)
nBasins = static_df.shape[0]  # count number of basins

# --- SAC-SMA Model ----------------------------------------------------

# grab all the model output from each basin
bcount = 0
#for basin in ['09306242']:#basin_list:
for basin in basin_list:

    # screen report
    bcount = bcount + 1
    print(f"Working on basin: {basin} ({bcount} of {len(basin_list)})")

    # pull observation column
    basin_df = master_dict[basin].filter(['qobs'])

    # loop through SCE seeds
    for sdex in range(len(SCE_SEEDS)):

        # load this seed's data and store in a basin-specific dataframe
        idx = np.where(meta_df['gauge_str'].values == basin)[0][0]
        fname = f"data/model_output_nldas/{meta_df.loc[idx, 'huc2']}/{meta_df.loc[idx, 'gauge_str']}_{SCE_SEEDS[sdex]}_model_output.txt"
        temp_df = pd.read_table(
            fname, delimiter='\s+', parse_dates={'date': [0, 1, 2]}, index_col='date')
        temp_df.rename(columns={'MOD_RUN': f"qsim_{sdex}"}, inplace=True)
        basin_df = pd.merge(
            basin_df, temp_df[f"qsim_{sdex}"], how='inner', left_index=True, right_index=True)

    # calculate the ensemble mean
    ensMean = basin_df.filter(regex='qsim_').mean(axis=1)
    basin_df.insert(0, 'qsim', ensMean)

    # store in a multi-basin dictionary
    if bcount == 1:
        sac_dict = {basin: basin_df}
    else:
        sac_dict.update({basin: basin_df})

# --- end of basin loop -----------------------------------------

# validation time mask
for basin in sac_dict:
    mask = (sac_dict[basin].index > val_start) & (sac_dict[basin].index <= val_end)
    sac_dict[basin] = sac_dict[basin][mask]

# save the ensemble results as a pickle
fname = "analysis/results_data/benchmark_sacsma_ensemble.pkl"
with open(fname, 'wb') as f:
    pickle.dump(sac_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# screen report
print('Finished reading calibrated SAC-SMA data.')

# --- National Water Model ---------------------------------------------

# load data
fname = 'data/nwm/nwm_daily.pkl'
with open(fname, 'rb') as handle:
    nwm_df = pickle.load(handle)

# load indexes
nwm_indexes = np.load('data/nwm/camels_id.npy')

# dimensions
assert len(nwm_df) == nBasins

# normalize by catchment area - this could likely be improved
areas = static_df['area_gages2']

bcount = 0
for basin in basin_list:

    # screen report
    bcount = bcount + 1
    print(f"Working on basin: {basin} ({bcount} of {len(basin_list)})")

    # find index in nwm list
    idx = np.where(int(basin) == nwm_indexes)[0][0]

    # pull observation column
    basin_df = master_dict[basin].filter(['qobs'])
    basin_df = pd.merge(
        basin_df, nwm_df[idx]['NWM_RUN'], how='inner', left_index=True, right_index=True)

    # normalize by catchment area
    a = basin_df['NWM_RUN'].values
    n = np.divide(a, (areas[idx] / 86.4))  # convert from m3/s to mm/d (factor = 86.4)
    basin_df['NWM_RUN'] = n
    basin_df.rename(columns={'NWM_RUN': f"qsim"}, inplace=True)

    # store in a multi-basin dictionary
    if bcount == 1:
        nwm_dict = {basin: basin_df}
    else:
        nwm_dict.update({basin: basin_df})

# validation time mask
for basin in nwm_dict:
    mask = (nwm_dict[basin].index > val_start) & (nwm_dict[basin].index <= val_end)
    nwm_dict[basin] = nwm_dict[basin][mask]

# save the ensemble results as a pickle
fname = "analysis/results_data/benchmark_nwm_retrospective.pkl"
with open(fname, 'wb') as f:
    pickle.dump(nwm_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# screen report
print('Finished reading National Water Model data.')
