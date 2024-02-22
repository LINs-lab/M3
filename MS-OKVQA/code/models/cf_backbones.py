import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loss import MultilabelCategoricalCrossEntropyLoss


class NCF(nn.Module):
    def __init__(
        self, hidden_dim, node_type_num, dropout, model_zoo_num, max_node_num, loss
    ):
        super(NCF, self).__init__()
        self.W_out_1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_out_2 = nn.Linear(hidden_dim, 1)
        self.W_fusion = nn.Linear(hidden_dim * 3, hidden_dim)

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
        node_embs = node_embs.resize(
            t_embs.size(0), self.max_node_num, node_embs.size(1)
        )
        node_embs = torch.mean(node_embs, dim=1)

        ## fusion
        final_embs = torch.cat((t_embs, v_embs, node_embs), dim=-1)
        final_embs = self.W_fusion(final_embs)

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
        node_embs = node_embs.reshape(-1, node_embs.size(-1))
        node_embs = self.layer_norm(node_embs)

        return node_embs


class NCFPlus(nn.Module):
    def __init__(
        self, hidden_dim, node_type_num, dropout, model_zoo_num, max_node_num, loss
    ):
        super(NCFPlus, self).__init__()
        self.W_out_1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_out_2 = nn.Linear(hidden_dim, 1)
        self.W_fusion = nn.Linear(hidden_dim * 4, hidden_dim)

        self.node2emb = nn.Embedding(node_type_num + 1, hidden_dim)
        self.t_proj = nn.Linear(768, hidden_dim)
        self.v_proj = nn.Linear(768, hidden_dim)
        self.p_proj = nn.Linear(768, hidden_dim)
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

    def forward(self, G, t, v, p, labels, is_train=True):
        ## t, v => proj => embeddings
        t_embs, v_embs, p_embs = (
            t.reshape(-1, 768),
            v.reshape(-1, 768),
            p.reshape(-1, 768),
        )
        t_embs, v_embs, p_embs = (
            self.t_proj(t_embs),
            self.v_proj(v_embs),
            self.p_proj(p_embs),
        )

        ## node embeddings
        node_embs = self.get_input_embeddings(G, t_embs)
        node_embs = node_embs.resize(
            t_embs.size(0), self.max_node_num, node_embs.size(1)
        )
        node_embs = torch.mean(node_embs, dim=1)

        ## fusion
        final_embs = torch.cat((t_embs, v_embs, p_embs, node_embs), dim=-1)
        final_embs = self.W_fusion(final_embs)

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

    def get_input_embeddings(self, G, t_embs):
        node_embs = self.node2emb(G.m_idx)
        node_embs = node_embs.reshape(t_embs.size(0), self.max_node_num, -1)
        node_embs = self.node_proj(node_embs)
        node_embs = node_embs.reshape(-1, node_embs.size(-1))
        node_embs = self.layer_norm(node_embs)

        return node_embs
