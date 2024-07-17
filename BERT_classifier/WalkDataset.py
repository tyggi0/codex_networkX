import torch
from torch.utils.data import Dataset, DataLoader


class WalkDataset(Dataset):
    def __init__(self, walks, labels):
        self.walks = walks
        self.labels = labels

    def __len__(self):
        return len(self.walks)

    def __getitem__(self, idx):
        return torch.tensor(self.walks[idx]), torch.tensor(self.labels[idx])


# Example usage
labels = [1] * 10  # Assuming all walks are valid for this example
dataset = WalkDataset(encoded_walks, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)