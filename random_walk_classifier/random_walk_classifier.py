import torch
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset, random_split
from codex.codex import Codex
from graph import Graph
from random_walk_generator import RandomWalkGenerator
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class RandomWalkClassifier:
    def __init__(self, device):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(self.device)

    @staticmethod
    def encode_walks(walks):
        return [' '.join(map(str, walk)) for walk in walks]

    def prepare_data(self, valid_walks, invalid_walks):
        encoded_valid_walks = self.encode_walks(valid_walks)
        encoded_invalid_walks = self.encode_walks(invalid_walks)

        labels_valid = [1] * len(encoded_valid_walks)
        labels_invalid = [0] * len(encoded_invalid_walks)

        walks = encoded_valid_walks + encoded_invalid_walks
        labels = labels_valid + labels_invalid

        print(f"Total encoded valid walks: {len(encoded_valid_walks)}")
        print(f"Total encoded invalid walks: {len(encoded_invalid_walks)}")
        print(f"Total combined walks: {len(walks)}")

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


def prepare_train_dataset(generator, classifier, num_walks, walk_length=6):
    valid_walks = generator.generate_random_walks(num_walks, walk_length)
    invalid_walks = generator.generate_invalid_random_walks(valid_walks)

    print("Valid Walks:")
    for i, walk in enumerate(valid_walks[:10]):  # Print first 10 valid walks
        print(f"Walk {i + 1}: {walk}")

    print("\nInvalid Walks:")
    for i, walk in enumerate(invalid_walks[:10]):  # Print first 10 invalid walks
        print(f"Walk {i + 1}: {walk}")

    walks, labels = classifier.prepare_data(valid_walks, invalid_walks)
    dataset = RandomWalkClassifier.WalkDataset(walks, labels, classifier.tokenizer)

    return dataset


def prepare_eval_dataset(classifier, codex, split):
    valid_triples = codex.split(split)
    invalid_triples = codex.split(f"{split}_negatives")

    valid_walks = valid_triples.values.tolist()
    invalid_walks = invalid_triples.values.tolist()

    print(f"\n{split.capitalize()} Valid Walks:")
    for i, walk in enumerate(valid_walks[:10]):  # Print first 10 valid walks
        print(f"Walk {i + 1}: {walk}")

    print(f"\n{split.capitalize()} Invalid Walks:")
    for i, walk in enumerate(invalid_walks[:10]):  # Print first 10 invalid walks
        print(f"Walk {i + 1}: {walk}")

    encoded_valid_walks = classifier.encode_walks(valid_walks)
    encoded_invalid_walks = classifier.encode_walks(invalid_walks)

    valid_labels = [1] * len(encoded_valid_walks)
    invalid_labels = [0] * len(encoded_invalid_walks)

    walks = encoded_valid_walks + encoded_invalid_walks
    labels = valid_labels + invalid_labels

    dataset = RandomWalkClassifier.WalkDataset(walks, labels, classifier.tokenizer)

    return dataset


def prepare_datasets(generator, classifier, codex, num_walks, walk_length=6):
    # Prepare training dataset from the graph
    train_dataset = prepare_train_dataset(generator, classifier, num_walks, walk_length)

    # Prepare validation dataset from Codex splits
    valid_dataset = prepare_eval_dataset(classifier, codex, "valid")

    # Prepare test dataset from Codex splits
    test_dataset = prepare_eval_dataset(classifier, codex, "test")

    return train_dataset, valid_dataset, test_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_score(torch.tensor(labels), predictions)  # TODO F1, confusion matrix (if necessry)
    return {"eval_accuracy": accuracy}


def tune_hyperparameters(trainer, train_dataset, valid_dataset):
    param_grid = {
        'learning_rate': [1e-5, 3e-5, 5e-5],
        'num_train_epochs': [2, 3, 4],
        'per_device_train_batch_size': [16, 32],  # Reduced batch sizes to avoid OutOfMemoryError: CUDA out of memory.
        # per_device_train_batch_size was [16, 32]
    }
    best_params = None
    best_score = float('-inf')

    for params in ParameterGrid(param_grid):
        print(f"Trying parameters: {params}")
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=params['learning_rate'],
            per_device_train_batch_size=params['per_device_train_batch_size'],
            per_device_eval_batch_size=32,
            num_train_epochs=params['num_train_epochs'],
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer.args = training_args
        trainer.train_dataset = train_dataset
        trainer.eval_dataset = valid_dataset

        trainer.train()
        eval_results = trainer.evaluate()

        print(f"Evaluation results: {eval_results}")

        if eval_results['eval_accuracy'] > best_score:
            best_score = eval_results['eval_accuracy']
            best_params = params

    print(f"Best parameters found: {best_params} with accuracy {best_score}")
    return best_params


def main(random_walk_name, tune, num_walks=4000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Codex
    codex = Codex(code="en", size="s")

    # Initialize and load the Graph
    graph = Graph(codex).load_graph()

    # Initialize classifier
    classifier = RandomWalkClassifier(device=device)

    # Initialize RandomWalkGenerator
    generator = RandomWalkGenerator(graph, random_walk_name)

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, valid_dataset, test_dataset = prepare_datasets(generator, classifier, codex, num_walks=num_walks)

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=classifier.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=classifier.tokenizer,
        data_collator=RandomWalkClassifier.WalkDataset.collate_fn,
        compute_metrics=compute_metrics
    )

    if tune:
        print("Tuning hyperparameters...")
        best_params = tune_hyperparameters(trainer, train_dataset, valid_dataset)
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=best_params['learning_rate'],
            per_device_train_batch_size=best_params['per_device_train_batch_size'],
            per_device_eval_batch_size=32,
            num_train_epochs=best_params['num_train_epochs'],
            weight_decay=0.01,
        )
        trainer.args = training_args

    # Train the model
    print("Training the model...")
    trainer.train()

    # Evaluate the model on test data
    print("Evaluating the model on test dataset...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Test dataset evaluation results: {test_results}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Random Walk Classifier')
    parser.add_argument('--random_walk', type=str, required=True,
                        help='Name of the random walk strategy (BrownianMotion, ERGRW)')
    parser.add_argument('--tune', action='store_true', help='Flag to tune hyperparameters')
    args = parser.parse_args()
    main(args.random_walk, args.tune)

    # Running script:
    # python random_walk_classifier/random_walk_classifier.py --random_walk <name_of_random_walk_strategy> [--tune]
    # python random_walk_classifier/random_walk_classifier.py --random_walk BrownianMotion
    # python random_walk_classifier/random_walk_classifier.py --random_walk BrownianMotion --tune
    # python random_walk_classifier/random_walk_classifier.py --random_walk ERGRW
    # python random_walk_classifier/random_walk_classifier.py --random_walk ERGRW --tune
