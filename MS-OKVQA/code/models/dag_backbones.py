import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from utils.loss import MultilabelCategoricalCrossEntropyLoss
from .transformer import TransformerEncoder


class GAT(nn.Module):
    def __init__(
        self, hidden_dim, node_type_num, dropout, model_zoo_num, max_node_num, loss
    ):
        super(GAT, self).__init__()
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=4)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1)
        self.W_start_node = nn.Linear(hidden_dim * 2, hidden_dim)

        self.W_out_1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_out_2 = nn.Linear(hidden_dim, 1)

        self.node2emb = nn.Embedding(node_type_num + 1, hidden_dim)
        self.t_proj = nn.Linear(768, hidden_dim)
        self.v_proj = nn.Linear(768, hidden_dim)
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

        self.model_zoo_num = model_zoo_num
        self.max_node_num = max_node_num

        self.loss = loss
        if loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss == "cce":
            self.criterion = MultilabelCategoricalCrossEntropyLoss()

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, G, t, v, labels, is_train=True):
        ## t, v => proj => embeddings
        t_embs, v_embs = t.reshape(-1, 768), v.reshape(-1, 768)
        t_embs, v_embs = self.t_proj(t_embs), self.v_proj(v_embs)

        ## node embeddings
        node_embs = self.get_input_embeddings(G, t_embs, v_embs)

        ## computation graph learner
        edge_index = G.edge_index
        node_embs = F.elu(self.conv1(node_embs, edge_index))
        node_embs = F.elu(self.conv2(node_embs, edge_index))

        ## aggregation
        final_embs = self.get_pooling_res(G, node_embs)
        final_embs = final_embs.reshape(-1, self.model_zoo_num, final_embs.size(-1))

        ## logits
        logits = self.W_out_2(F.leaky_relu(self.W_out_1(final_embs)))
        logits = logits.reshape(-1, self.model_zoo_num)

        if is_train:
            labels = labels.reshape(-1, self.model_zoo_num)
            if self.loss == "bce":
                loss = self.criterion(logits, labels)
            elif self.loss == "cce":
                loss = torch.mean(self.criterion(labels, logits))

            return loss
        else:
            return logits

    def get_pooling_res(self, G, x):
        # reshape
        mask = (G.m_idx >= 1).float()
        x = x.reshape(-1, self.max_node_num, x.size(-1))
        mask = mask.reshape(-1, self.max_node_num, 1)
        # pooling
        weighted_x = x * mask
        sum_weighted_x = torch.sum(weighted_x, dim=(1))
        sum_mask = torch.sum(mask, dim=(1))
        result = sum_weighted_x / sum_mask

        return result

    def get_input_embeddings(self, G, t_embs, v_embs):
        node_embs = self.node2emb(G.m_idx)
        node_embs = node_embs.reshape(t_embs.size(0), self.max_node_num, -1)
        node_embs = self.node_proj(node_embs)
        node_embs[:, 0] = self.W_start_node(torch.cat((v_embs, t_embs), dim=-1))
        node_embs = node_embs.reshape(-1, node_embs.size(-1))
        node_embs = self.layer_norm(node_embs)

        return node_embs


class GRU(nn.Module):
    def __init__(
        self, hidden_dim, node_type_num, dropout, model_zoo_num, max_node_num, loss
    ):
        super(GRU, self).__init__()
        # Reset gate weights
        self.Wr = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.Ur = nn.Linear(hidden_dim, hidden_dim)

        # Update gate weights
        self.Wz = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.Uz = nn.Linear(hidden_dim, hidden_dim)

        # Candidate activation weights
        self.W = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)

        self.W_out_1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_out_2 = nn.Linear(hidden_dim, 1)
        self.W_start_node = nn.Linear(hidden_dim * 2, hidden_dim)

        self.node2emb = nn.Embedding(node_type_num + 1, hidden_dim)
        self.t_proj = nn.Linear(768, hidden_dim)
        self.v_proj = nn.Linear(768, hidden_dim)
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim

        self.model_zoo_num = model_zoo_num
        self.max_node_num = max_node_num

        self.loss = loss
        if loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss == "cce":
            self.criterion = MultilabelCategoricalCrossEntropyLoss()

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, G, t, v, labels, is_train=True):
        ## t, v => proj => embeddings
        t_embs, v_embs = t.reshape(-1, 768), v.reshape(-1, 768)
        t_embs, v_embs = self.t_proj(t_embs), self.v_proj(v_embs)

        ## node embeddings
        node_embs = self.get_input_embeddings(G, t_embs, v_embs)
        node_embs = node_embs.resize(
            t_embs.size(0), self.max_node_num, node_embs.size(1)
        )

        ## computation graph learner
        edge_index = G.edge_index
        edge_index = edge_index.cpu().numpy()
        batch_ids = edge_index[0, :] // self.max_node_num
        masks = torch.zeros(
            (t_embs.size(0), self.max_node_num, self.max_node_num),
            dtype=torch.float,
            device=node_embs.device,
        )

        for idx in range(edge_index.shape[1]):
            batch_id = batch_ids[idx]
            s, e = (
                edge_index[0, idx] % self.max_node_num,
                edge_index[1, idx] % self.max_node_num,
            )
            masks[batch_id, e, s] = 1

        ## forward
        batch = node_embs.size(0)
        states = torch.zeros(batch, self.max_node_num, self.hidden_dim).to(
            node_embs.device
        )
        for i in range(self.max_node_num):
            x_i = node_embs[:, i, :].view(batch, -1)
            mask = masks[:, i, :].unsqueeze(1)
            h_i = torch.bmm(mask, states).squeeze(1)

            ## GRU forward
            combined = torch.cat((x_i, h_i), dim=1)
            r = torch.sigmoid(self.Wr(combined) + self.Ur(h_i))
            z = torch.sigmoid(self.Wz(combined) + self.Uz(h_i))
            h_tilde = torch.tanh(self.W(combined) + r * self.U(h_i))
            h_new = (1 - z) * h_i + z * h_tilde

            ## store
            states[:, i, :] = h_new

        ## get final state
        matrix = torch.sum(masks, dim=2)
        last_positive_index = (matrix > 0).sum(dim=1).long()
        last_positive_index = last_positive_index
        final_states = states[
            torch.arange(states.size(0)), last_positive_index.squeeze()
        ]

        ## logits
        logits = self.W_out_2(F.leaky_relu(self.W_out_1(final_states)))
        logits = logits.reshape(-1, self.model_zoo_num)

        if is_train:
            labels = labels.reshape(-1, self.model_zoo_num)
            if self.loss == "bce":
                loss = self.criterion(logits, labels)
            elif self.loss == "cce":
                loss = torch.mean(self.criterion(labels, logits))

            return loss
        else:
            return logits

    def get_input_embeddings(self, G, t_embs, v_embs):
        node_embs = self.node2emb(G.m_idx)
        node_embs = node_embs.reshape(t_embs.size(0), self.max_node_num, -1)
        node_embs = self.node_proj(node_embs)
        node_embs[:, 0] = self.W_start_node(torch.cat((v_embs, t_embs), dim=-1))
        node_embs = node_embs.reshape(-1, node_embs.size(-1))
        node_embs = self.layer_norm(node_embs)

        return node_embs


class Transformer(nn.Module):
    def __init__(
        self, hidden_dim, node_type_num, dropout, model_zoo_num, max_node_num, loss
    ):
        super(Transformer, self).__init__()
        # Reset gate weights
        self.Wr = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.Ur = nn.Linear(hidden_dim, hidden_dim)

        # Update gate weights
        self.Wz = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.Uz = nn.Linear(hidden_dim, hidden_dim)

        # Candidate activation weights
        self.W = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)

        self.W_out_1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_out_2 = nn.Linear(hidden_dim, 1)
        self.W_start_node = nn.Linear(hidden_dim * 2, hidden_dim)

        self.node2emb = nn.Embedding(node_type_num + 1, hidden_dim)
        self.t_proj = nn.Linear(768, hidden_dim)
        self.v_proj = nn.Linear(768, hidden_dim)
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim

        self.model_zoo_num = model_zoo_num
        self.max_node_num = max_node_num

        self.transformer = TransformerEncoder(
            n_position=max_node_num,
            d_model=hidden_dim,
            n_heads=1,
            dropout=dropout,
            n_layers=1,
        )

        self.loss = loss
        if loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss == "cce":
            self.criterion = MultilabelCategoricalCrossEntropyLoss()

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                try:
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
                except:
                    pass

    def forward(self, G, t, v, labels, is_train=True):
        ## t, v => proj => embeddings
        t_embs, v_embs = t.reshape(-1, 768), v.reshape(-1, 768)
        t_embs, v_embs = self.t_proj(t_embs), self.v_proj(v_embs)  # [B, H]

        ## node embeddings
        node_embs = self.get_input_embeddings(G, t_embs, v_embs)
        node_embs = node_embs.resize(
            t_embs.size(0), self.max_node_num, node_embs.size(1)
        )  # [B, N, H]

        ## computation graph learner
        edge_index = G.edge_index
        edge_index = edge_index.cpu().numpy()
        batch_ids = edge_index[0, :] // self.max_node_num
        masks = torch.zeros(
            (t_embs.size(0), self.max_node_num, self.max_node_num),
            dtype=torch.float,
            device=node_embs.device,
        )  # [B, N, N]

        for idx in range(edge_index.shape[1]):
            batch_id = batch_ids[idx]
            s, e = (
                edge_index[0, idx] % self.max_node_num,
                edge_index[1, idx] % self.max_node_num,
            )
            masks[batch_id, e, s] = 1

        att_masks = (1.0 - masks) * -10000.0

        ## forward
        node_embs = self.transformer(node_embs, att_masks)

        ## get final state
        final_embs = torch.mean(node_embs, axis=1)

        ## logits
        logits = self.W_out_2(F.leaky_relu(self.W_out_1(final_embs)))
        logits = logits.reshape(-1, self.model_zoo_num)

        if is_train:
            labels = labels.reshape(-1, self.model_zoo_num)
            if self.loss == "bce":
                loss = self.criterion(logits, labels)
            elif self.loss == "cce":
                loss = torch.mean(self.criterion(labels, logits))

            return loss
        else:
            return logits

    def get_input_embeddings(self, G, t_embs, v_embs):
        node_embs = self.node2emb(G.m_idx)
        node_embs = node_embs.reshape(t_embs.size(0), self.max_node_num, -1)
        node_embs = self.node_proj(node_embs)
        node_embs[:, 0] = self.W_start_node(torch.cat((v_embs, t_embs), dim=-1))
        node_embs = node_embs.reshape(-1, node_embs.size(-1))
        node_embs = self.layer_norm(node_embs)

        return node_embs
