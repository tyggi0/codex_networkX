import json
import os
import random

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from transformers.trainer_utils import get_last_checkpoint

from walk_dataset import WalkDataset


class ModelTrainer:
    def __init__(self, output_dir, classifier, tune, train_dataset, valid_dataset, test_dataset, device):
        self.output_dir = output_dir
        self.tune = tune
        self.classifier = classifier
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.best_params = None

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        predictions = np.argmax(logits, axis=-1)
        eval_accuracy = accuracy_score(labels, predictions)
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        roc_auc = roc_auc_score(labels, probabilities)

        # Generate the classification report
        class_report = classification_report(labels, predictions, output_dict=True)

        # Generate the confusion matrix
        conf_matrix = confusion_matrix(labels, predictions)

        return {
            "eval_accuracy": eval_accuracy,
            "roc_auc": roc_auc,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix
        }

    def get_optimizer(self, args):
        lr = args.learning_rate if args else self.best_params['learning_rate']
        return Adam(self.classifier.model.parameters(), lr=lr)

    @staticmethod
    def get_scheduler(optimizer, num_training_steps, warmup_steps):
        # Use a linear learning rate scheduler with warmup
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

    def get_trainer(self, args, batch_size, total_steps, warmup_steps):
        train_loader, valid_loader, _ = self.create_data_loaders(batch_size)

        optimizer = self.get_optimizer(args)  # Pass the correct learning rate from `args`

        lr_scheduler = self.get_scheduler(optimizer, num_training_steps=total_steps, warmup_steps=warmup_steps)

        return Trainer(
            model=self.classifier.model,
            args=args,
            train_dataset=train_loader.dataset,
            eval_dataset=valid_loader.dataset,
            tokenizer=self.classifier.tokenizer,
            data_collator=WalkDataset.collate_fn,
            compute_metrics=self.compute_metrics,
            optimizers=(optimizer, lr_scheduler)  # Pass the optimizer and scheduler
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

    def tune_hyperparameters(self, n_iter=5):
        param_distributions = {
            'learning_rate': np.logspace(-5, -4, num=100),  # 1e-5 to 1e-4
            'num_train_epochs': [3, 4, 5, 6],
            'per_device_train_batch_size': [16, 32],
            'warmup_ratio': [0.0, 0.1, 0.2]
        }
        best_params = None
        best_score = float('-inf')

        completed_trials = set()

        # Load completed trials from a file if it exists
        completed_trials_file = f"{self.output_dir}/completed_trials.json"
        if os.path.exists(completed_trials_file):
            with open(completed_trials_file, 'r') as f:
                completed_trials = set(json.load(f))

        for _ in range(n_iter):
            # Randomly sample hyperparameters
            params = {
                'learning_rate': random.choice(param_distributions['learning_rate']),
                'num_train_epochs': random.choice(param_distributions['num_train_epochs']),
                'per_device_train_batch_size': random.choice(param_distributions['per_device_train_batch_size']),
                'warmup_ratio': random.choice(param_distributions['warmup_ratio']),
            }

            params_tuple = tuple(params.items())

            if params_tuple in completed_trials:
                print(f"Skipping completed parameters: {params}")
                continue

            batch_size = params['per_device_train_batch_size']
            total_steps = len(self.train_dataset) // batch_size * params['num_train_epochs']
            warmup_steps = int(total_steps * params['warmup_ratio'])

            print(f"Trying parameters: {params} with warmup_steps={warmup_steps}")

            # Create a directory name that includes the chosen parameters
            output_dir_name = (f"{self.output_dir}/fine_tuning_lr{params['learning_rate']}_"
                               f"epochs{params['num_train_epochs']}_"
                               f"batch{params['per_device_train_batch_size']}_"
                               f"warmup{params['warmup_ratio']}")

            training_args = TrainingArguments(
                output_dir=output_dir_name,
                evaluation_strategy='epoch',  # Evaluate at the end of each epoch
                save_strategy='epoch',  # Save a checkpoint at the end of each epoch
                save_total_limit=10,  # Only keep the last 10 checkpoints
                learning_rate=params['learning_rate'],
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=params['num_train_epochs'],
                weight_decay=0.01,
                warmup_steps=warmup_steps,
                logging_strategy="epoch",  # ELog training data stats for loss
                logging_dir=f'{output_dir_name}/logs',
                load_best_model_at_end=True,  # Load the best model at the end of training
                metric_for_best_model="eval_accuracy",  # Specify the metric to use for selecting the best model
                greater_is_better=True,  # Specify that higher metric values are better
            )

            trainer = self.get_trainer(training_args, batch_size, total_steps, warmup_steps)

            # Check if there's a checkpoint available to resume from
            last_checkpoint = None
            if os.path.exists(output_dir_name):
                last_checkpoint = get_last_checkpoint(output_dir_name)

            trainer.train(resume_from_checkpoint=last_checkpoint)

            eval_results = trainer.evaluate()

            print(f"Evaluation results: {eval_results}")

            if eval_results['eval_accuracy'] > best_score:
                best_score = eval_results['eval_accuracy']
                best_params = params

            # Mark this set of parameters as completed
            completed_trials.add(params_tuple)
            with open(completed_trials_file, 'w') as f:
                json.dump(list(completed_trials), f)

        self.best_params = best_params

        # Save the best parameters using json.dump
        with open(f'{self.output_dir}/best_hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=4)

        print(f"Best parameters found: {best_params} with accuracy {best_score}")
        return best_params

    def train(self):
        if self.tune:
            # Path to the best hyperparameters file
            best_params_file = os.path.join(self.output_dir, 'best_hyperparameters.json')

            if os.path.exists(best_params_file):
                # Load the best hyperparameters from the file if it exists
                print("Loading best hyperparameters from file...")
                with open(best_params_file, 'r') as f:
                    self.best_params = json.load(f)
            else:
                # If the file doesn't exist, tune the hyperparameters
                print("Tuning hyperparameters...")
                self.tune_hyperparameters()
        else:
            # If not tuning, set default best_params
            self.best_params = {
                'learning_rate': 2e-5,
                'num_train_epochs': 5,
                'per_device_train_batch_size': 16,
                'warmup_ratio': 0.1
            }

        batch_size = self.best_params['per_device_train_batch_size']
        total_steps = len(self.train_dataset) // batch_size * self.best_params['num_train_epochs']
        warmup_steps = int(self.best_params.get('warmup_ratio', 0.1) * total_steps)

        # Create a directory name that includes the chosen parameters
        output_dir_name = (f"{self.output_dir}/training_lr{self.best_params['learning_rate']}_"
                           f"epochs{self.best_params['num_train_epochs']}_"
                           f"batch{self.best_params['per_device_train_batch_size']}_"
                           f"warmup{self.best_params['warmup_ratio']}")

        training_args = TrainingArguments(
            output_dir=output_dir_name,
            evaluation_strategy='epoch',  # Evaluate at the end of each epoch
            save_strategy='epoch',  # Save a checkpoint at the end of each epoch
            learning_rate=self.best_params['learning_rate'],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=self.best_params['num_train_epochs'],
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            logging_dir=f'{output_dir_name}/logs',
            logging_strategy="epoch",
            load_best_model_at_end=True,  # Load the best model at the end of training
            metric_for_best_model="eval_accuracy",  # Specify the metric to use for selecting the best model
            greater_is_better=True,  # Specify that higher metric values are better
        )

        # Initialize early stopping callback
        early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

        train_loader, valid_loader, test_loader = self.create_data_loaders(batch_size)

        trainer = self.get_trainer(training_args, batch_size, total_steps, warmup_steps)

        # Add the early stopping callback
        trainer.add_callback(early_stopping)

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
