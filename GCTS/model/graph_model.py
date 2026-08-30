import torch
from torch.nn import functional as func
import torch.nn.functional as F
from torch import nn
from torch_geometric import nn as gnn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import  GATConv
import torch_scatter


class MLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_units=32, num_layers=1, bias_term = True):
        super(MLP, self).__init__()
        if num_layers == 1:
            self.layers = nn.Linear(num_features, num_classes, bias = bias_term)
        elif num_layers > 1:
            layers = [nn.Linear(num_features, hidden_units, bias = bias_term),
                      #nn.BatchNorm1d(hidden_units),
                      nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_units, hidden_units, bias = bias_term),
                               nn.ReLU()])
            layers.append(nn.Linear(hidden_units, num_classes, bias = bias_term))
            self.layers = nn.Sequential(*layers)
        else:
            raise ValueError()

    def forward(self, x):
        return self.layers(x)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, X, A_norm):
        return torch.mm(A_norm, self.linear(X))


class GNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout=args.dropout
        self.gcn1 = GCNLayer(args.input_dim, args.hidden_dim)
        self.gcn_mu = GCNLayer(args.hidden_dim, args.output_dim)
        self.gcn_logvar = GCNLayer(args.hidden_dim, args.output_dim)

    def forward(self, X, A_norm):
        H = F.relu(self.gcn1(X, A_norm))
        H = F.dropout(H, p=self.dropout, training=self.training)
        mu = self.gcn_mu(H, A_norm)
        logvar = self.gcn_logvar(H, A_norm)
        logvar = torch.clamp(logvar, min=-8, max=8)
        return mu, logvar
    

def reparameterize(mu, logvar, training=True):
    if training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    return mu


#Not used anymore !
class GIN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_units=32, decoder_out_dim = 128,num_layers=3, dropout=0.15,
                 mlp_layers=2, train_eps=False, is_encoder = True):
        super(GIN, self).__init__()
        convs, bns = [], []
        for i in range(num_layers):
            input_dim = num_features if i == 0 else hidden_units
            if is_encoder : 
                hidden_dim = hidden_units
            else : 
                hidden_dim = hidden_units if i != num_layers - 1 else decoder_out_dim
            convs.append(gnn.GINConv(MLP(input_dim, hidden_dim, hidden_dim, mlp_layers),
                                     train_eps=train_eps))
            bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(p = self.dropout)
        self.is_encoder = is_encoder
        if self.is_encoder != True : # Add learnable mask parameters
            self.encoder_mask = torch.nn.Parameter(torch.zeros(decoder_out_dim))
            self.decoder_mask = torch.nn.Parameter(torch.zeros(int(num_features)))
        self.final_layers = torch.nn.Linear(hidden_dim, hidden_dim)
            

    def forward(self, data): # Do not consider batchwise training now
        x = data.x 
        edge_index = data.edge_index
        
        if self.is_encoder :
            h_list = [x]
            for conv, bn in zip(self.convs, self.bns):
                h = conv(h_list[-1], edge_index)
                h = self.dropout_layer(h)
                h = torch.relu(h)
                h_list.append(h)
            out = torch.cat(h_list[1:], 1)
            return out
        else : 
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = self.dropout_layer(x)
                x = torch.relu(x)
            x = self.final_layers(x)
            return x
        


class GAT_GNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.heads = 4
        self.dropout = args.dropout
        
        # GAT takes dense A via edge_index, not A_norm
        self.gat1 = GATConv(
            args.input_dim, 
            args.hidden_dim // self.heads,
            heads=self.heads, 
            dropout=args.dropout,
            concat=True
        )
        self.gat_mu = GATConv(
            args.hidden_dim, args.output_dim,
            heads=1, concat=False
        )
        self.gat_logvar = GATConv(
            args.hidden_dim, args.output_dim,
            heads=1, concat=False
        )

    def forward(self, X, edge_index):
        H = F.elu(self.gat1(X, edge_index))
        H = F.dropout(H, p=self.dropout, training=self.training)
        mu     = self.gat_mu(H, edge_index)
        logvar = torch.clamp(self.gat_logvar(H, edge_index), -8, 8)
        return mu, logvar


class MLP_Decoder(nn.Module) : 
    def __init__(self, in_dim, hid_dim, out_dim) :
        super(MLP_Decoder, self).__init__()
        
        self.lin1 = torch.nn.Linear(in_dim, hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim, out_dim)
        
    def forward(self, x) : 
        
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        
        return x
    
