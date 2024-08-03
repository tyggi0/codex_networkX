import torch
from torch.utils.data import Dataset


class WalkDataset(Dataset):
    def __init__(self, walks, labels, tokenizer):
        self.walks = walks
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.walks)

    def __getitem__(self, idx):
        walk = self.walks[idx]
        label = self.labels[idx]
        walk_str = ' '.join(map(str, walk))
        encoding = self.tokenizer(walk_str, return_tensors='pt', padding='max_length', truncation=True,
                                  max_length=512)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_masks = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }
