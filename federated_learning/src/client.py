"""
FederatedClient — handles local training and validation for one FL client.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm


class FederatedClient:
    """
    Represents one hospital / data silo in the federation.

    Typical round:
        client.set_model(global_model)
        client.set_optimizer(lr)
        weights, losses, accs = client.train_local(epochs)
        val_loss, val_acc = client.validate()
    """

    def __init__(self, client_id, train_loader, val_loader, device="cuda"):
        self.client_id    = client_id
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.model        = None
        self.optimizer    = None
        self.criterion    = nn.CrossEntropyLoss()
        self.dataset_size = len(train_loader.dataset)

    # ----- model / optimiser --------------------------------------------------

    def set_model(self, global_model):
        """Deep-copy the server's global model."""
        self.model = copy.deepcopy(global_model)
        self.model.to(self.device)

    def set_optimizer(self, learning_rate=0.001):
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    # ----- local training -----------------------------------------------------

    def train_local(self, epochs=5, verbose=False):
        """
        Run local training on client data.

        Returns
        -------
        state_dict, list[float], list[float]
        """
        self.model.train()
        epoch_losses, epoch_accs = [], []

        for ep in range(epochs):
            running_loss = 0.0
            correct = total = 0

            pbar = tqdm(self.train_loader,
                        desc=f"{self.client_id} Epoch {ep+1}/{epochs}",
                        disable=not verbose, leave=False)

            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted  = outputs.max(1)
                total   += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if verbose:
                    pbar.set_postfix(Loss=f"{running_loss/(batch_idx+1):.4f}",
                                    Acc=f"{100.*correct/total:.2f}%")

            ep_loss = running_loss / len(self.train_loader)
            ep_acc  = 100.0 * correct / total
            epoch_losses.append(ep_loss)
            epoch_accs.append(ep_acc)

            if verbose:
                print(f"  {self.client_id} E{ep+1}: Loss={ep_loss:.4f}  Acc={ep_acc:.2f}%")

        return self.model.state_dict(), epoch_losses, epoch_accs

    # ----- validation ---------------------------------------------------------

    def validate(self):
        """Evaluate on the local validation set. Returns (loss, accuracy%)."""
        self.model.eval()
        val_loss = 0.0
        correct = total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total   += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return val_loss / len(self.val_loader), 100.0 * correct / total

    # ----- helpers ------------------------------------------------------------

    def get_dataset_size(self):
        return self.dataset_size
