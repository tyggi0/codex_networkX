import json

import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score

from walk_dataset import WalkDataset


class ModelTrainer:
    def __init__(self, output_dir, classifier, tune, train_dataset, valid_dataset, test_dataset, device, batch_size):
        self.output_dir = output_dir
        self.tune = tune
        self.classifier = classifier
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.batch_size = batch_size
        self.best_params = None

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(logits, dim=-1)

        accuracy = accuracy_score(labels, predictions.numpy())
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions.numpy(), average='binary')
        probabilities = torch.softmax(logits, dim=-1)[:, 1].numpy()  # ROC needs probabilities
        roc_auc = roc_auc_score(labels, probabilities)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }

    def get_trainer(self, args, train_dataset, eval_dataset):
        return Trainer(
            model=self.classifier.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.classifier.tokenizer,
            data_collator=WalkDataset.collate_fn,
            compute_metrics=self.compute_metrics
        )

    def create_data_loaders(self, batch_size):
        # Prepare data loaders from datasets
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=WalkDataset.collate_fn)
        valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=WalkDataset.collate_fn)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=WalkDataset.collate_fn)
        return train_loader, valid_loader, test_loader

    def tune_hyperparameters(self):
        param_grid = {
            'learning_rate': [1e-5, 3e-5, 5e-5],
            'num_train_epochs': [3, 4, 5],
            'per_device_train_batch_size': [16, 32],
        }
        best_params = None
        best_score = float('-inf')

        for params in ParameterGrid(param_grid):
            print(f"Trying parameters: {params}")
            batch_size = params['per_device_train_batch_size']
            train_loader, valid_loader, _ = self.create_data_loaders(batch_size)

            training_args = TrainingArguments(
                output_dir=f'{self.output_dir}/fine_tuning',
                evaluation_strategy='epoch',  # Evaluate at the end of each epoch
                save_strategy='epoch',  # Save a checkpoint at the end of each epoch
                # save_total_limit=10,  # Only keep the last 10 checkpoints
                learning_rate=params['learning_rate'],
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=params['num_train_epochs'],
                weight_decay=0.01,
                logging_strategy="epoch",  # ELog training data stats for loss
                logging_dir=f'{self.output_dir}/fine_tuning/logs',
            )

            trainer = self.get_trainer(training_args, train_loader.dataset, valid_loader.dataset)
            trainer.train()
            eval_results = trainer.evaluate()

            print(f"Evaluation results: {eval_results}")

            if eval_results['eval_accuracy'] > best_score:
                best_score = eval_results['eval_accuracy']
                best_params = params

        self.best_params = best_params

        # Save the best parameters using json.dump
        with open(f'{self.output_dir}/best_hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=4)

        print(f"Best parameters found: {best_params} with accuracy {best_score}")
        return best_params

    def train(self):
        if self.tune:
            print("Tuning hyperparameters...")
            self.tune_hyperparameters()
            batch_size = self.best_params['per_device_train_batch_size']
            training_args = TrainingArguments(
                output_dir=f"{self.output_dir}/training",
                evaluation_strategy='epoch',  # Evaluate at the end of each epoch
                save_strategy='epoch',  # Save a checkpoint at the end of each epoch
                learning_rate=self.best_params['learning_rate'],
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=self.best_params['num_train_epochs'],
                weight_decay=0.01,
                logging_dir=f'{self.output_dir}/training/logs',
                logging_strategy="epoch"
            )
        else:
            batch_size = self.batch_size
            training_args = TrainingArguments(
                output_dir=f"{self.output_dir}/training",
                evaluation_strategy="epoch",
                save_strategy='epoch',  # Save a checkpoint at the end of each epoch
                learning_rate=2e-5,  # Got 0.5 accurary with 1e-4
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=5,  # was 3
                weight_decay=0.01,
                logging_dir=f'{self.output_dir}/training/logs',
                logging_strategy="epoch",
                # warmup_steps=500,  # Adding warmup steps
            )

        train_loader, valid_loader, test_loader = self.create_data_loaders(batch_size)

        trainer = self.get_trainer(training_args, train_loader.dataset, valid_loader.dataset)

        # Train the model
        print("Training the model...")
        trainer.train()

        # Save final model
        trainer.save_model(f'{self.output_dir}/final_model')
        self.classifier.tokenizer.save_pretrained(f'{self.output_dir}/final_model')

        # Perform evaluation on test data
        print("Evaluating the model on the test dataset...")
        test_results = self.evaluate(test_loader)

        # Save final evaluation results using json.dump
        with open(f"{self.output_dir}/final_evaluation_results.json", "w") as f:
            json.dump(test_results, f, indent=4)

        print(f"Evaluation results on test dataset: {test_results}")
        return test_results

    def evaluate(self, data_loader):
        self.classifier.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.classifier.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = self.compute_metrics((all_logits, all_labels))
        return metrics
