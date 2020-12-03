import torch
import numpy as np
import scipy.sparse as sp

def scipy2torch(adj):
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i,v, torch.Size(adj.shape))

def load_data(data, feature_type):
    adj_full = scipy2torch(sp.load_npz('data/' + data + '/adj_full.npz'))
    adj_self = scipy2torch(sp.load_npz('data/' + data + '/adj_self.npz'))
    adj_futr = scipy2torch(sp.load_npz('data/' + data + '/adj_futr.npz'))
    adj_past = scipy2torch(sp.load_npz('data/' + data + '/adj_past.npz'))
    features = torch.from_numpy(np.load('data/' + data + '/features_' + feature_type + '.npy'))
    label = torch.from_numpy(np.load('data/' + data + '/label.npy'))
    rr = torch.from_numpy(np.load('data/' + data + '/role.npy'))
    role = dict()
    role['tr'] = np.where(rr == 0)[0]
    role['te'] = np.where(rr == 2)[0]
    return adj_full, adj_self, adj_futr, adj_past, features, label, role