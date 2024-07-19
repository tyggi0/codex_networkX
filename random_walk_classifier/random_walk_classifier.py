import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import networkx as nx

from random_walk.brownian_motion_random_walk import BrownianMotionRandomWalk
from random_walk.ergrw_random_walk import ERGRWRandomWalk


class RandomWalkClassifier:
    def __init__(self, graph: nx.Graph, random_walk_strategy, device):
        self.graph = graph
        self.random_walk = self.get_random_walk_strategy(random_walk_strategy)
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)

    def get_random_walk_strategy(self, name):
        if name == "BrownianMotion":
            self.random_walk = BrownianMotionRandomWalk(self, )
        elif name == "ERGRW":
            self.random_walk = ERGRWRandomWalk(self)
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

        data = list(zip(walks, labels))
        random.shuffle(data)
        walks, labels = zip(*data)

        return walks, labels

    class WalkDataset(Dataset):
        def __init__(self, walks, labels):
            self.walks = walks
            self.labels = labels

        def __len__(self):
            return len(self.walks)

        def __getitem__(self, idx):
            return torch.tensor(self.walks[idx]), torch.tensor(self.labels[idx])

    def train(self, dataloader, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                inputs, labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                print(f'Epoch {epoch}, Loss: {loss.item()}')

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy: {correct / total * 100}%')
