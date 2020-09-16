import torch
from torch import nn
import torch.nn.functional as F


class RGCN_Layer(nn.Module):
    """ A Relation GCN module operated on documents graphs. """

    def __init__(self, params, in_dim, mem_dim, num_layers, relation_cnt=5):
        super().__init__()
        self.params = params
        self.layers = num_layers
        self.device = torch.device("cuda" if params['gpu'] != -1 else "cpu")
        self.mem_dim = mem_dim
        self.relation_cnt = relation_cnt
        self.in_dim = in_dim

        self.in_drop = nn.Dropout(params['gcn_in_drop'])
        self.gcn_drop = nn.Dropout(params['gcn_out_drop'])

        # gcn layer
        self.W_0 = nn.ModuleList()
        self.W_r = nn.ModuleList()
        # for i in range(self.relation_cnt):
        for i in range(relation_cnt):
            self.W_r.append(nn.ModuleList())

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W_0.append(nn.Linear(input_dim, self.mem_dim).to(self.device))
            for W in self.W_r:
                W.append(nn.Linear(input_dim, self.mem_dim).to(self.device))


    def forward(self, nodes, adj, section):
        """

        :param nodes:  batch_size * node_size * node_emb
        :param adj:  batch_size * 5 * node_size * node_size
        :param section:  (Tensor <B, 3>) #entities/#mentions/#sentences per batch
        :return:
        """
        gcn_inputs = self.in_drop(nodes)

        maskss = []
        denomss = []
        for batch in range(adj.shape[0]):
            masks = []
            denoms = []
            for i in range(self.relation_cnt):
                denom = torch.sparse.sum(adj[batch, i], dim=1).to_dense()
                t_g = denom + torch.sparse.sum(adj[batch, i], dim=0).to_dense()
                mask = t_g.eq(0)
                denoms.append(denom.unsqueeze(1))
                masks.append(mask)
            denoms = torch.sum(torch.stack(denoms), 0)
            denoms = denoms + 1
            masks = sum(masks)
            maskss.append(masks)
            denomss.append(denoms)
        denomss = torch.stack(denomss) # 40 * 61 * 1

        # sparse rgcn layer
        for l in range(self.layers):
            gAxWs = []
            for j in range(self.relation_cnt):
                gAxW = []

                bxW = self.W_r[j][l](gcn_inputs)
                for batch in range(adj.shape[0]):

                    xW = bxW[batch]  # 255 * 25
                    AxW = torch.sparse.mm(adj[batch][j], xW)  # 255, 25
                    # AxW = AxW/ denomss[batch][j]  # 255, 25
                    gAxW.append(AxW)
                gAxW = torch.stack(gAxW)
                gAxWs.append(gAxW)
            gAxWs = torch.stack(gAxWs, dim=1)
            # print("denomss", denomss.shape)
            # print((torch.sum(gAxWs, 1) + self.W_0[l](gcn_inputs)).shape)
            gAxWs = F.relu((torch.sum(gAxWs, 1) + self.W_0[l](gcn_inputs)) / denomss)  # self loop
            gcn_inputs = self.gcn_drop(gAxWs) if l < self.layers - 1 else gAxWs

        return gcn_inputs, maskss
