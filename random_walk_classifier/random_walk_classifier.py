import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import networkx as nx
import sys
import random
from brownian_motion_random_walk import BrownianMotionRandomWalk
from ergrw_random_walk import ERGRWRandomWalk


class RandomWalkClassifier:
    def __init__(self, graph: nx.Graph, random_walk_strategy, device):
        self.graph = graph
        self.random_walk = self.get_random_walk_strategy(random_walk_strategy)
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)  # Updated optimizer

    def get_random_walk_strategy(self, name):
        if name == "BrownianMotion":
            return BrownianMotionRandomWalk(self.graph)
        elif name == "ERGRW":
            return ERGRWRandomWalk(self.graph)
        else:
            raise ValueError(f"Unknown random walk strategy: {name}")

    def generate_random_walks(self, num_walks, walk_length):
        return self.random_walk.generate_walks(num_walks, walk_length)

    def generate_invalid_random_walks(self, num_walks, walk_length):
        nodes = list(self.graph.nodes)
        invalid_walks = []
        for _ in range(num_walks):
            walk = []
            current_node = random.choice(nodes)
            for _ in range(walk_length):
                walk.append(current_node)
                if random.random() > 0.5:
                    current_node = random.choice(nodes)
                else:
                    neighbors = list(self.graph.neighbors(current_node))
                    if neighbors:
                        current_node = random.choice(neighbors)
                    else:
                        break
            invalid_walks.append(walk)
        return invalid_walks

    def encode_walks(self, walks):
        return [self.tokenizer.encode(' '.join(map(str, walk)), add_special_tokens=True) for walk in walks]

    def prepare_data(self, valid_walks, invalid_walks):
        encoded_valid_walks = self.encode_walks(valid_walks)
        encoded_invalid_walks = self.encode_walks(invalid_walks)

        labels_valid = [1] * len(encoded_valid_walks)
        labels_invalid = [0] * len(encoded_invalid_walks)

        walks = encoded_valid_walks + encoded_invalid_walks
        labels = labels_valid + labels_invalid

        return walks, labels

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
            encoding = self.tokenizer(walk, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
            return (encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze(),
                    torch.tensor(label, dtype=torch.long))

        @staticmethod
        def collate_fn(batch):
            input_ids = torch.stack([item[0] for item in batch])
            attention_masks = torch.stack([item[1] for item in batch])
            labels = torch.stack([item[2] for item in batch])
            return input_ids, attention_masks, labels

        def get_dataloader(self, batch_size=8, shuffle=False):
            return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    def train(self, dataloader, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                inputs, attention_masks, labels = [item.to(self.device) for item in batch]
                self.optimizer.zero_grad()
                outputs = self.model(inputs, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                sys.stdout.write(f'Epoch {epoch}, Loss: {loss.item()}\n')

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, attention_masks, labels = [item.to(self.device) for item in batch]
                outputs = self.model(inputs, attention_mask=attention_masks)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        sys.stdout.write(f'Accuracy: {correct / total * 100}%\n')