import torch
import numpy as np
import scipy.sparse as sp
import numpy as np
from sklearn import metrics

def calc_f1(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    # y_pred = np.argmax(y_pred, axis=1)
    return metrics.f1_score(y_true, y_pred, average="weighted")

def calc_class_level_acc(y_true, y_pred, num_cls):
    y_true = np.argmax(y_true, axis=1)
    f1 = metrics.f1_score(y_true, y_pred, average=None)
    acc = list()
    for i in range(num_cls):
        acc.append(metrics.accuracy_score(y_true[y_true == i], y_pred[y_true == i]))
    acc = np.array(acc)
    return acc, f1

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
    label = torch.from_numpy(np.load('data/' + data + '/label.npy')).type(torch.LongTensor)
    rr = torch.from_numpy(np.load('data/' + data + '/role.npy'))
    role = dict()
    role['tr'] = np.where(rr == 0)[0]
    role['te'] = np.where(rr == 2)[0]
    return adj_full, adj_self, adj_futr, adj_past, features, label, role
