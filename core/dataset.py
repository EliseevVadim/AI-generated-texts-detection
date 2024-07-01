import torch
from torch.utils.data import Dataset


class AITextDetectionDataset(Dataset):
    def __init__(self, X, y, tokenizer=None, device='cpu', max_length=512):
        self.X = tokenizer(X.tolist(),
                           truncation=True,
                           padding=True,
                           return_tensors="pt",
                           max_length=max_length).to(device)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.int64).to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X.input_ids[idx], self.X.attention_mask[idx], self.y[idx]
