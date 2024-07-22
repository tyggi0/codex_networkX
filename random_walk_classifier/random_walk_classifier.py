import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, TrainingArguments, Trainer
from datasets import Dataset as HFDataset, load_metric
import networkx as nx
import sys
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from evaluate import load

from brownian_motion_random_walk import BrownianMotionRandomWalk
from ergrw_random_walk import ERGRWRandomWalk


class RandomWalkClassifier:
    def __init__(self, graph: nx.Graph, random_walk_strategy, device):
        self.graph = graph
        self.random_walk = self.get_random_walk_strategy(random_walk_strategy)
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.metric = load('accuracy', trust_remote_code=True)

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
        return [self.tokenizer(' '.join(map(str, walk)), truncation=True, padding='max_length', max_length=512) for walk
                in walks]

    def prepare_data(self, valid_walks, invalid_walks):
        encoded_valid_walks = self.encode_walks(valid_walks)
        encoded_invalid_walks = self.encode_walks(invalid_walks)

        labels_valid = [1] * len(encoded_valid_walks)
        labels_invalid = [0] * len(encoded_invalid_walks)

        walks = encoded_valid_walks + encoded_invalid_walks
        labels = labels_valid + labels_invalid

        return HFDataset.from_dict({'walks': walks, 'labels': labels})

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def tune_hyperparameters(self, train_dataset, valid_dataset, num_samples=10, max_num_epochs=10, gpus_per_trial=1):
        def model_init():
            return BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        def encode(examples):
            outputs = self.tokenizer(
                examples['walks'], truncation=True, padding='max_length', max_length=512
            )
            return outputs

        encoded_train_dataset = train_dataset.map(encode, batched=True)
        encoded_valid_dataset = valid_dataset.map(encode, batched=True)

        training_args = TrainingArguments(
            "test",
            evaluation_strategy="steps",
            eval_steps=500,
            disable_tqdm=True,
            num_train_epochs=max_num_epochs,
        )

        trainer = Trainer(
            args=training_args,
            train_dataset=encoded_train_dataset,
            eval_dataset=encoded_valid_dataset,
            tokenizer=self.tokenizer,
            model_init=model_init,
            compute_metrics=self.compute_metrics,
        )

        search_space = {
            "learning_rate": tune.loguniform(1e-6, 1e-4),
            "batch_size": tune.choice([8, 16, 32]),
            "epochs": tune.randint(1, 5)
        }

        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=10,
            grace_period=1,
            reduction_factor=2
        )

        reporter = CLIReporter(
            metric_columns=["accuracy", "training_iteration"]
        )

        ray.init()
        analysis = trainer.hyperparameter_search(
            direction="maximize",
            backend="ray",
            n_trials=num_samples,
            search_alg=search_space,
            scheduler=scheduler,
            keep_checkpoints_num=1,
            checkpoint_score_attr="training_iteration",
            resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
            progress_reporter=reporter,
            local_dir="./ray_results",
            name="tune_transformer"
        )
        ray.shutdown()

        best_trial = analysis.get_best_trial("accuracy", "max", "last")
        best_config = best_trial.config

        print(f"Best trial config: {best_config}")
        print(f"Best trial final validation accuracy: {analysis.best_result['accuracy']}")

        return best_config

    def set_hyperparameters(self, learning_rate, batch_size, epochs):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def get_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

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
