"""
QPSOServer — Quantum-behaved Particle Swarm Optimisation for FL aggregation.

Each client is a "particle".  The server maintains:
  • personal best (pbest)  — best state_dict each client has ever produced.
  • global  best (gbest)   — best state_dict across all clients.
  • mean    best (mbest)   — element-wise mean of all pbests.

Position update (per parameter tensor):
    φ  ~ U(0,1)    u ~ U(0,1)
    p  = φ·mean(pbests) + (1−φ)·gbest          # attraction point
    x' = p  ±  β · |mbest − x| · ln(1/u)       # QPSO step

β (contraction-expansion coefficient) controls exploration vs exploitation.
"""

import copy
import torch
import torch.nn as nn


class QPSOServer:
    """Central server using QPSO-based aggregation."""

    def __init__(self, global_model, clients, device="cuda", beta=0.7):
        self.global_model = global_model
        self.clients      = clients
        self.device       = device
        self.beta         = beta

        # per-client personal bests
        self.personal_best        = {}          # client_id → state_dict
        self.personal_best_scores = {}          # client_id → float (val acc)

        # global best
        self.global_best       = None           # state_dict
        self.global_best_score = 0.0

        # mean best (computed each round)
        self.mean_best = None

        print(f"QPSO Server  |  clients={len(clients)}  β={beta}")

    # ----- initialisation -----------------------------------------------------

    def initialize_particles(self):
        """Set every pbest and gbest to the current global model."""
        state = copy.deepcopy(self.global_model.state_dict())
        for c in self.clients:
            self.personal_best[c.client_id]        = copy.deepcopy(state)
            self.personal_best_scores[c.client_id] = 0.0
        self.global_best       = copy.deepcopy(state)
        self.global_best_score = 0.0
        print("✅ QPSO particles initialised")

    # ----- pbest / gbest updates ----------------------------------------------

    def update_personal_best(self, client_id, weights, val_acc):
        if val_acc > self.personal_best_scores[client_id]:
            self.personal_best[client_id]        = copy.deepcopy(weights)
            self.personal_best_scores[client_id] = val_acc
            return True
        return False

    def update_global_best(self, client_id, val_acc):
        if val_acc > self.global_best_score:
            self.global_best       = copy.deepcopy(self.personal_best[client_id])
            self.global_best_score = val_acc
            return True
        return False

    # ----- mbest ---------------------------------------------------------------

    def calculate_mean_best(self):
        """Element-wise mean of all personal-best state_dicts."""
        first_id = self.clients[0].client_id
        self.mean_best = copy.deepcopy(self.personal_best[first_id])

        for k in self.mean_best:
            self.mean_best[k] = torch.zeros_like(self.mean_best[k],
                                                 dtype=torch.float32)
        for c in self.clients:
            pb = self.personal_best[c.client_id]
            for k in self.mean_best:
                self.mean_best[k] += pb[k].float()

        n = len(self.clients)
        for k in self.mean_best:
            self.mean_best[k] /= n

    # ----- QPSO aggregation ---------------------------------------------------

    def qpso_aggregate(self, client_weights_list):
        """
        Perform one QPSO aggregation step.

        Parameters
        ----------
        client_weights_list : list[(client_id, state_dict, val_acc)]

        Returns
        -------
        aggregated state_dict
        """
        # 1. update personal / global bests
        for cid, w, acc in client_weights_list:
            pb = self.update_personal_best(cid, w, acc)
            gb = self.update_global_best(cid, acc)
            if pb:
                print(f"  {cid}: pbest ↑ {acc:.2f}%")
            if gb:
                print(f"  {cid}: gbest ↑ {acc:.2f}%")

        # 2. mean best
        self.calculate_mean_best()

        # 3. QPSO position update for every parameter key
        agg = copy.deepcopy(self.global_best)

        for k in agg:
            # sum of personal bests  (used to get mean pbest for attraction)
            pbest_sum = torch.zeros_like(agg[k], dtype=torch.float32)
            for c in self.clients:
                pbest_sum += self.personal_best[c.client_id][k].float()

            phi  = torch.rand_like(agg[k].float())
            u    = torch.rand_like(agg[k].float())

            # attraction point
            p = phi * (pbest_sum / len(self.clients)) \
              + (1 - phi) * self.global_best[k].float()

            # sign ±1
            sign = torch.where(
                torch.rand_like(agg[k].float()) < 0.5,
                torch.ones_like(agg[k].float()),
               -torch.ones_like(agg[k].float()),
            )

            mbest   = self.mean_best[k].float()
            current = agg[k].float()

            # QPSO step
            new_val = p + sign * self.beta \
                      * torch.abs(mbest - current) \
                      * torch.log(1.0 / (u + 1e-8))

            agg[k] = new_val.to(self.global_best[k].dtype)

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
