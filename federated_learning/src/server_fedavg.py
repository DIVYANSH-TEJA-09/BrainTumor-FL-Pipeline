"""
FedAvgServer — Federated Averaging aggregation.

Aggregation rule
    w_global = Σ (n_k / n_total) · w_k
where n_k is the training-set size of client k.
"""

import copy
import torch
import torch.nn as nn


class FedAvgServer:
    """Central server implementing the FedAvg algorithm."""

    def __init__(self, global_model, clients, device="cuda"):
        self.global_model = global_model
        self.clients      = clients
        self.device       = device
        self.total_samples = sum(c.get_dataset_size() for c in clients)

        print(f"FedAvg Server  |  clients={len(clients)}  "
              f"total_samples={self.total_samples}")

    # ----- aggregation --------------------------------------------------------

    def aggregate_weights(self, client_weights):
        """
        Weighted average of client state_dicts.

        Parameters
        ----------
        client_weights : list[(state_dict, dataset_size)]

        Returns
        -------
        aggregated state_dict
        """
        first_sd = client_weights[0][0]
        agg = {}

        for k in first_sd:
            if first_sd[k].is_floating_point():
                # Weighted average for float params
                accumulated = torch.zeros_like(first_sd[k], dtype=torch.float32)
                for state_dict, n_k in client_weights:
                    w = n_k / self.total_samples
                    accumulated = accumulated + state_dict[k].float() * w
                agg[k] = accumulated.to(first_sd[k].dtype)
            else:
                # Non-float (e.g. num_batches_tracked): copy from first client
                agg[k] = first_sd[k].clone()

        return agg

    # ----- evaluation ---------------------------------------------------------

    def evaluate_global_model(self, test_loader):
        """Return (accuracy%, loss) on a test DataLoader."""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                total_loss += criterion(outputs, labels).item()
                _, pred = outputs.max(1)
                total   += labels.size(0)
                correct += pred.eq(labels).sum().item()

        return 100.0 * correct / total, total_loss / len(test_loader)

    # ----- helpers ------------------------------------------------------------

    def get_global_model(self):
        return copy.deepcopy(self.global_model)
