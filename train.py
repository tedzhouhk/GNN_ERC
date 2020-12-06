import numpy
import argparse
import torch
import scipy.sparse as sp
from utils import *
from modules import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to train.')

    parser.add_argument('-d', type=str, help='which dataset to parse')
    parser.add_argument('-f', type=str, help='which feature type to use (pooled or mean)')
    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('-r', type=float, help='dropout')
    parser.add_argument('--dims', nargs='+', type=int, help='dims for part of the aggregated feature')
    parser.add_argument('-l', nargs='+', type=int, help='num layers of each sub-gnn module')
    parser.add_argument('-e', type=int, help='number of epochs to train')
    parser.add_argument('-s', type=int, default=-1, help='seed of random number generator')
    parser.add_argument('-v', type=float, default=0.2, help='percentage ')

    args = parser.parse_args()

    adj_full, adj_self, adj_futr, adj_past, features, label, role = load_data(args.d, args.f)

    if args.s != -1:
        torch.manual_seed(args.s)
        np.random.seed(args.s)
    model = ERCGNN(args.l, features.shape[1], args.dims, torch.max(label).item() + 1, [adj_full, adj_self, adj_futr, adj_past], dropout=args.r)

    if not args.no_cuda:
        # adj_full = adj_full.cuda()
        # adj_self = adj_self.cuda()
        # adj_futr = adj_futr.cuda()
        # adj_past = adj_past.cuda()
        features = features.cuda()
        label = label.cuda()
        model = model.cuda()
    
    # import pdb; pdb.set_trace()

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters())
    split = int(role['tr'].shape[0] * (1 - args.v))

    best_e = 0
    best_val = 0
    best_test = 0
    best_pred = None
    for e in range(args.e):
        model.train()
        optimizer.zero_grad()
        pred = model(features)
        np.random.shuffle(role['tr'])
        # cross validation
        train_nodes = role['tr'][:split]
        val_nodes = role['tr'][split:]
        loss = loss_fn(pred[train_nodes], label[train_nodes]).sum()
        loss.backward()
        optimizer.step()
        train_acc = calc_f1(label[train_nodes].detach().cpu().numpy(), pred[train_nodes].detach().cpu().numpy())
        model.eval()
        pred = model(features)
        val_acc = calc_f1(label[val_nodes].detach().cpu().numpy(), pred[val_nodes].detach().cpu().numpy())
        test_acc = calc_f1(label[role['te']].detach().cpu().numpy(), pred[role['te']].detach().cpu().numpy())
        if val_acc > best_val:
            best_e = e
            best_val = val_acc
            best_test = test_acc
            best_pred = pred[role['te']].detach().cpu().numpy()
        print('epoch: {} train f1weighted: {:.3f} val f1weighted: {:.3f} (test f1weighted: {:.3f})'.format(e, train_acc, val_acc, test_acc))
    print('Optimization Finished!')
    print('Epoch: {}, test f1weighted: {:.3f}, test acc: {:.3f}'.format(best_e, best_test, calc_acc(label[role['te']].detach().cpu().numpy(), best_pred)))
    f1micro, f1macro = calc_f1mima(label[role['te']].detach().cpu().numpy(), best_pred)
    print('\t test f1micro: {:.3f}, test f1macro: {:.3f}'.format(f1micro, f1macro))

    if args.d == 'IEMOCAP':
        cat = ['hap','sad','neu','ang','exc','fru']
    elif args.d == 'MELD':
        cat = ['neu','sup','fea','sad','joy','dis','ang']
    elif args.d == 'dailydialogue':
        cat = ['neu','ang','dis','fea','hap','sad','sur']
    accs, f1s = calc_class_level_acc(label[role['te']].detach().cpu().numpy(), best_pred, torch.max(label).item() + 1)
    print('Class Level Acc and F1:')
    for name, acc, f1 in zip(cat, accs, f1s):
        print('{}: acc: {:.3f} f1: {:.3f}'.format(name, acc, f1))
    # print('Avg acc: {:.3f} f1: {:.3f}'.format(np.mean(accs), np.mean(f1s)))
    