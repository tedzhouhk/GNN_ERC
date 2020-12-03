import numpy
import argparse
import torch
import scipy.sparase as sp
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to train.')

    parser.add_argument('-d', type=str, help='which dataset to parse')
    parser.add_argument('-f', type=str, help='which feature type to use (pooled or mean)')
    parser.add_argument('--no_cuda', type=bool, default=False, action='store_true')
    parser.add_argument('-r', type=float, help='dropout')
    parser.add_argument('--dims', nargs='+', type=int, help='dims for part of the aggregated feature')
    parser.add_argument('-l', nargs='+', type=int, help='num layers of each sub-gnn module')

    args = parser.parse_args()

    adj_full, adj_self, adj_futr, adj_past, features, label, role = load_data(args.d, args.f)
    if not args.no_cuda:
        adj_full = adj_full.cuda()
        adj_self = adj_self.cuda()
        adj_futr = adj_futr.cuda()
        adj_past = adj_past.cuda()
        features = features.cuda()
        label = label.cuda()
    
