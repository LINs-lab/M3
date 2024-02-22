from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        graph, t_embs, v_embs = self.data_list[index]
        return graph, t_embs, v_embs

    def __len__(self):
        return len(self.data_list)


class CustomDatasetP(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        graph, t_embs, v_embs, p_embs, time = self.data_list[index]
        return graph, t_embs, v_embs, p_embs, time

    def __len__(self):
        return len(self.data_list)
