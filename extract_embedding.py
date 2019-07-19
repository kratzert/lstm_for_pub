"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., Nearing, G., "Prediction 
in Ungauged Basins with Long Short-Term Memory Networks". submitted to Water Resources Research 
(2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import pdb
import pickle

import numpy as np
import torch

# load the trained model
fname = 'runs/run_0307_2317_seed184127/model_epoch30.pt'
model = torch.load(fname)

sh = model['lstm.weight_sh'].cpu().data.numpy()
print(sh)
np.savetxt('embedding_weights.txt', sh)

pdb.set_trace()

# extract weights from the embedding layer
weights0 = model['lstm.embedder.net.0.weight'].data.numpy()
weights1 = model['lstm.embedder.net.1.weight'].data.numpy()
biases0 = model['lstm.embedder.net.0.bias'].data.numpy()
biases1 = model['lstm.embedder.net.1.bias'].data.numpy()

np.savetxt('weights0.txt', weights0)
np.savetxt('weights1.txt', weights1)
np.savetxt('biases0.txt', biases0)
np.savetxt('biases1.txt', biases1)

fname = 'runs/run_0307_1808_seed27666/usgs_id_to_int.p'
with open(fname, 'rb') as f:
    usgs_id_to_int = pickle.load(f)

with open('usgs_id_to_int.csv', 'w') as f:
    for key in usgs_id_to_int.keys():
        f.write("%s,%s\n" % (key, usgs_id_to_int[key]))
