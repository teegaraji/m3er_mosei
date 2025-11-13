import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from m3er_model import M3ER, EarlyStopping


class MOSEIDataset(Dataset):
    """Dataset class untuk CMU-MOSEI"""

    def __init__(self, data_dict):
        self.speech = torch.FloatTensor(data_dict["speech"])
        self.text = torch.FloatTensor(data_dict["text"])
        self.visual = torch.FloatTensor(data_dict["visual"])
        self.labels = torch.LongTensor(data_dict["labels"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "speech": self.speech[idx],
            "text": self.text[idx],
            "visual": self.visual[idx],
            "labels": self.labels[idx],
        }


class M3ERTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load data
        self.load_data()

        # Create model
        self.create_model()

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-5),
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", 15), min_delta=0.001
        )

        # History
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],
        }

    def load_data(self):
        """Load processed data"""
        data_dir = self.config["data_dir"]

        print("\nLoading data...")
        with open(os.path.join(data_dir, "train_data.pkl"), "rb") as f:
            train_data = pickle.load(f)

        with open(os.path.join(data_dir, "val_data.pkl"), "rb") as f:
            val_data = pickle.load(f)

        with open(os.path.join(data_dir, "test_data.pkl"), "rb") as f:
            test_data = pickle.load(f)

        with open(os.path.join(data_dir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        print(f"✓ Train samples: {len(train_data['labels'])}")
        print(f"✓ Val samples: {len(val_data['labels'])}")
        print(f"✓ Test samples: {len(test_data['labels'])}")

        # Create datasets
        self.train_dataset = MOSEIDataset(train_data)
        self.val_dataset = MOSEIDataset(val_data)
        self.test_dataset = MOSEIDataset(test_data)

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.metadata = metadata

    def create_model(self):
        """Create M3ER model"""
        model_config = {
            "timestep": self.metadata["timestep"],
            "feature_dims": {
                "speech": self.metadata["speech_dim"],
                "text": self.metadata["text_dim"],
                "visual": self.metadata["visual_dim"],
            },
            "n_classes": self.metadata["n_classes"],
            "beta": self.config.get("beta", 1.0),
        }

        from m3er_model import M3ER

        self.model = M3ER(model_config).to(self.device)

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"\n{'='*60}")
        print("Model Architecture")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"{'='*60}\n")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')

        for batch_idx, batch in enumerate(pbar):
            speech = batch["speech"].to(self.device)
            text = batch["text"].to(self.device)
            visual = batch["visual"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(speech, text, visual)
            loss, loss_dict = self.model.compute_loss(outputs, labels)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs["final_pred"], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.2f}%"}
            )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                speech = batch["speech"].to(self.device)
                text = batch["text"].to(self.device)
                visual = batch["visual"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(speech, text, visual)
                loss, _ = self.model.compute_loss(outputs, labels)

                total_loss += loss.item()

                _, predicted = torch.max(outputs["final_pred"], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return avg_loss, accuracy, f1

    def train(self):
        """Full training loop"""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        best_val_acc = 0
        best_epoch = 0

        for epoch in range(self.config["epochs"]):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_f1 = self.validate()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

            # Print epoch summary
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config['epochs']} Summary")
            print(f"{'='*60}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}"
            )
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'='*60}\n")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                self.save_checkpoint("best_model.pth", epoch, val_acc)
                print(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)\n")

            # Save latest checkpoint
            self.save_checkpoint("latest_model.pth", epoch, val_acc)

            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(
                    f"Best model was at epoch {best_epoch} with Val Acc: {best_val_acc:.2f}%"
                )
                print(f"{'='*60}\n")
                break

        print(f"\n{'='*60}")
        print("Training Completed!")
        print(f"{'='*60}")
        print(f"Best Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
        print(f"{'='*60}\n")

        return best_val_acc

    def test(self, checkpoint_path="best_model.pth"):
        """Test model"""
        print("\n" + "=" * 60)
        print("Testing Model")
        print("=" * 60)

        # Load best model
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Loaded model from epoch {checkpoint['epoch']}")

        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                speech = batch["speech"].to(self.device)
                text = batch["text"].to(self.device)
                visual = batch["visual"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(speech, text, visual)

                _, predicted = torch.max(outputs["final_pred"], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        f1 = f1_score(all_labels, all_preds, average="weighted")

        # Classification report
        class_names = self.metadata["class_names"]
        report = classification_report(
            all_labels, all_preds, target_names=class_names, digits=4
        )

        print(f"\n{'='*60}")
        print("Test Results")
        print(f"{'='*60}")
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"\n{report}")
        print(f"{'='*60}\n")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm, class_names)

        return accuracy, f1, cm

    def save_checkpoint(self, filename, epoch, val_acc):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "config": self.config,
        }
        torch.save(checkpoint, filename)

    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        print("✓ Confusion matrix saved to confusion_matrix.png")

    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        axes[0].plot(self.history["train_loss"], label="Train Loss")
        axes[0].plot(self.history["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy plot
        axes[1].plot(self.history["train_acc"], label="Train Acc")
        axes[1].plot(self.history["val_acc"], label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=300, bbox_inches="tight")
        print("✓ Training history saved to training_history.png")


# Main execution
if __name__ == "__main__":
    # Configuration
    config = {
        "data_dir": "/kaggle/input/mosei-preprocess",
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "epochs": 100,
        "beta": 1.0,
        "early_stopping_patience": 15,
    }

    # Create trainer
    trainer = M3ERTrainer(config)

    # Train
    best_val_acc = trainer.train()

    # Plot training history
    trainer.plot_training_history()

    # Test
    test_acc, test_f1, cm = trainer.test()

    # Save results
    results = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "config": config,
        "history": trainer.history,
    }

    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n✓ All results saved!")
