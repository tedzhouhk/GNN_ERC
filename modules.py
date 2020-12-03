import torch
import numpy as np

class Perceptron(torch.nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0, norm=True, act=True):
        super(Perceptron, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(in_dim, out_dim))
        torch.nn.init.xavier_uniform_(self.weight.data)
        self.bias = torch.nn.Parameter(torch.empty(out_dim))
        torch.nn.init.zeros_(self.bias.data)
        if norm:
            self.norm = torch.nn.BatchNorm1d(out_dim, eps=1e-9, tracking_running_stats=True)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, f_in):
        f_in = self.dropout(f_in)
        f_in = torch.mm(f_in, self.weight) + self.bias
        if act:
            f_in = torch.nn.functional.relu(f_in)
            f_in = self.norm[i](f_in)
        return f_in


class VANILA_GCN(torch.nn.Module):

    def __init__(self, num_layer, in_dim, out_dim, adj, dropout=0, norm=True):
        super(VANILA_GCN, self).__init__()
        self.num_layer = num_layer
        self.register_buffer('adj', adj)
        self.weight = list()
        self.bias = list()
        self.weight.append(torch.nn.Parameter(torch.empty(in_dim, out_dim)))
        torch.nn.init.xavier_uniform_(self.weight[-1].data)
        self.bias.append(torch.nn.Parameter(torch.empty(out_dim)))
        torch.nn.init.zeros_(self.bias[-1].data)
        for _ in range(1, num_layer):
            self.weight.append(torch.nn.Parameter(torch.empty(out_dim, out_dim)))
            torch.nn.init.xavier_uniform_(self.weight[-1].data)
            self.bias.append(torch.nn.Parameter(torch.empty(out_dim)))
            torch.nn.init.zeros_(self.bias[-1].data)
        self.norm = list()
        if norm:
            for _ in range(num_layer):
                self.norm.append(torch.nn.BatchNorm1d(out_dim, eps=1e-9, racking_running_stats=True))
        self.dropout=list()
        for _ in range(num_layer):
            self.dropout.append(torch.nn.Dropout(p=dropout))

    def forward(self, f_in):
        for i in range(self.num_layer):
            f_in = self.dropout[i](f_in)
            f_in = torch.mm(f_in, self.weight[i]) + self.bias[i]
            f_in = torch.sparse.mm(self.adj, f_in)
            f_in = torch.nn.functional.relu(f_in)
            f_in = self.norm[i](f_in)
        return f_in

class GAT(torch.nn.module):

    def __init__(self, num_layer, in_dim, out_dim, adj, mulhead=8, dropout=0, norm=True):
        super(GAT, self).__init__()
        self.num_layer = num_layer
        self.mulhead = mulhead
        self.register_buffer('adj', adj)
        self.weight_self = list()
        self.bias_self = list()
        self.weight_neigh = list()
        self.bias_neigh = list()
        self.attention_self = list()
        self.attention_neigh = list()
        for _ in range(self.mulhead):
            self.weight_self.append(list())
            self.bias_self.append(list())
            self.weight_neigh.append(list())
            self.bias_neigh.append(list())
            self.weight_self[-1].append(torch.nn.Parameter(torch.empty(in_dim, out_dim / mulhead)))
            torch.nn.init.xavier_uniform_(self.weight_self[-1][-1].data)
            self.bias_self[-1].append(torch.nn.Parameter(torch.empty(out_dim / mulhead)))
            torch.nn.init.zeros_(self.bias_self[-1][-1].data)
            for _ in range(1, num_layer):
                self.weight_self[-1].append(torch.nn.Parameter(torch.empty(out_dim, out_dim / mulhead)))
                torch.nn.init.xavier_uniform_(self.weight_self[-1][-1].data)
                self.bias_self[-1].append(torch.nn.Parameter(torch.empty(out_dim / mulhead)))
                torch.nn.init.zeros_(self.bias_self[-1][-1].data)
            self.weight_neigh[-1].append(torch.nn.Parameter(torch.empty(in_dim, out_dim / mulhead)))
            torch.nn.init.xavier_uniform_(self.weight_neigh[-1][-1].data)
            self.bias_neigh[-1].append(torch.nn.Parameter(torch.empty(out_dim / mulhead)))
            torch.nn.init.zeros_(self.bias_neigh[-1][-1].data)
            for _ in range(1, num_layer):
                self.weight_neigh[-1].append(torch.nn.Parameter(torch.empty(out_dim, out_dim / mulhead)))
                torch.nn.init.xavier_uniform_(self.weight_neigh[-1][-1].data)
                self.bias_neigh[-1].append(torch.nn.Parameter(torch.empty(out_dim / mulhead)))
                torch.nn.init.zeros_(self.bias_neigh[-1][-1].data)
            self.attention_self.append(list())
            self.attention_neigh.append(list())
            for _ in range(num_layer):
                self.attention_self[-1].append(torch.nn.Parameter(torch.empty(out_dim / mulhead, 1)))
                torch.nn.init.ones_(self.ones_(self.attention_self[-1][-1].data))
                self.attention_neigh[-1].append(torch.nn.Parameter(torch.empty(out_dim / mulhead, 1)))
                torch.nn.init.ones_(self.ones_(self.attention_neigh[-1][-1].data))
        self.norm = list()
        if norm:
            for _ in range(num_layer):
                self.norm.append(torch.nn.BatchNorm1d(out_dim, eps=1e-9, racking_running_stats=True))
        self.dropout=list()
        for _ in range(num_layer):
            self.dropout.append(torch.nn.Dropout(p=dropout))

    def forward(self, f_in):
        for i in range(self.num_layer):
            f_in = self.dropout[i](f_in)
            f_out = list()
            for j in range(self.mulhead):
                f_self = torch.mm(f_in, self.weight_self[j][i]) + self.bias_self[i]
                f_self = torch.nn.functional.relu(f_self)
                attention_self = torch.nn.functional.leaky_relu(torch.mm(f_self, self.attention_self[j][i]), negative_slope=0.2)
                f_neigh = torch.mm(f_in, self.weight_neigh[j][i]) + self.bias_neigh[i]
                f_neigh = torch.nn.functional.relu(f_neigh)
                attention_neigh = torch.nn.functional.leaky_relu(torch.mm(f_self, self.attention_neigh[j][i]), negative_slope=0.2)
                attention_norm = (attention_self[self.adj._indices()[0]] + attention_neigh[self.adj._indices()[1]]) * adj._values()
                att_adj = torch.sparse.FloatTensor(self.adj._indices(), attention_norm, torch.Size(self.adj.shape))
                f_out.append(torch.sparse.mm(att_adj, f_neigh))
            f_in = torch.cat(f_out, 1)
            f_in = self.norm[i](f_in)
        return f_in

class ERCGNN(torch.nn.Module):

    def __init__(self, num_layer, in_dim, out_dims, num_class, adjs, dropout=0.0):
        super(ERCGNN, self).__init__()
        self.fcgat = GAT(num_layer[0], in_dim, out_dims[0], adjs[0], dropout=dropout)
        self.uttrgnn = VANILA_GCN(num_layer[1], in_dim, out_dims[1], adjs[1], dropout=dropout)
        self.pastgnn = VANILA_GCN(num_layer[2], in_dim, out_dims[2], adjs[2], dropout=dropout)
        self.futrgnn = VANILA_GCN(num_layer[3], in_dim, out_dims[3], adjs[3], dropout=dropout)
        self.selfgnn = Perceptron(in_dim, out_dims[4], dropout=dropout)
        self.classifier = Perceptron(np.sum(out_dims), num_class, dropout=dropout, act=False)

    def forward(self,f_in):
        f_out = list()
        f_out.append(self.fcgat(f_in))
        f_out.append(self.uttrgnn(f_in))
        f_out.append(self.pastgnn(f_in))
        f_out.append(self.futrgnn(f_in))
        f_out.append(self.selfgnn(f_in))
        f_out = torch.cat(f_out, 1)
        f_out = self.classifier(f_out)
        return f_out