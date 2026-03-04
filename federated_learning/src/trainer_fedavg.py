"""
FedAvg training loop — runs N communication rounds.
"""

import time
import torch
import pandas as pd


def train_fedavg(
    server,
    clients,
    global_test_loader,
    num_rounds=100,
    local_epochs=5,
    learning_rate=0.001,
    save_every=10,
    save_dir="/kaggle/working",
    verbose=True,
):
    """
    Full FedAvg training.

    Returns
    -------
    history : dict  (round, global_test_acc, global_test_loss,
                     clientN_val_acc, round_time)
    """
    print("=" * 80)
    print("  FEDAVG TRAINING START")
    print("=" * 80)

    history = {
        "round": [],
        "global_test_acc": [],
        "global_test_loss": [],
        "client1_val_acc": [],
        "client2_val_acc": [],
        "client3_val_acc": [],
        "round_time": [],
    }

    best_acc = 0.0

    for rnd in range(1, num_rounds + 1):
        t0 = time.time()
        print(f"\n{'='*70}  ROUND {rnd}/{num_rounds}  {'='*6}")

        client_weights = []

        for client in clients:
            print(f"  {client.client_id} training …", end=" ")
            client.set_model(server.get_global_model())
            client.set_optimizer(learning_rate)
            w, _, _ = client.train_local(epochs=local_epochs, verbose=verbose)
            val_loss, val_acc = client.validate()
            print(f"val_acc={val_acc:.2f}%")

            history[f"{client.client_id}_val_acc"].append(val_acc)
            client_weights.append((w, client.get_dataset_size()))

        # aggregate
        agg = server.aggregate_weights(client_weights)
        server.global_model.load_state_dict(agg)

        # evaluate global
        g_acc, g_loss = server.evaluate_global_model(global_test_loader)
        dt = time.time() - t0

        history["round"].append(rnd)
        history["global_test_acc"].append(g_acc)
        history["global_test_loss"].append(g_loss)
        history["round_time"].append(dt)

        print(f"  Global  acc={g_acc:.2f}%  loss={g_loss:.4f}  ({dt:.1f}s)")

        # best model
        if g_acc > best_acc:
            best_acc = g_acc
            torch.save(server.global_model.state_dict(),
                       f"{save_dir}/models/fedavg_best.pth")
            print(f"  ✅ New best! {best_acc:.2f}%")

        # checkpoint
        if rnd % save_every == 0:
            torch.save(server.global_model.state_dict(),
                       f"{save_dir}/models/fedavg_round_{rnd}.pth")

        # incremental CSV
        pd.DataFrame(history).to_csv(
            f"{save_dir}/results/fedavg/metrics.csv", index=False)

    print("\n" + "=" * 80)
    print(f"  FEDAVG DONE — best accuracy {best_acc:.2f}%")
    print("=" * 80)
    return history
