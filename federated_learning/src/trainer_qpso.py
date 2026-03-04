"""
QPSO-FL training loop — runs N communication rounds.
"""

import time
import torch
import pandas as pd


def train_qpso(
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
    Full QPSO-FL training.

    Returns
    -------
    history : dict  (round, global_test_acc/loss, global_best_score,
                     clientN_val_acc, clientN_pbest_score, round_time)
    """
    print("=" * 80)
    print("  QPSO-FL TRAINING START")
    print("=" * 80)

    server.initialize_particles()

    history = {
        "round": [],
        "global_test_acc": [],
        "global_test_loss": [],
        "global_best_score": [],
        "client1_val_acc": [],
        "client2_val_acc": [],
        "client3_val_acc": [],
        "client1_pbest_score": [],
        "client2_pbest_score": [],
        "client3_pbest_score": [],
        "round_time": [],
    }

    best_acc = 0.0

    for rnd in range(1, num_rounds + 1):
        t0 = time.time()
        print(f"\n{'='*70}  ROUND {rnd}/{num_rounds}  {'='*6}")

        cw_list = []

        for client in clients:
            print(f"  {client.client_id} training …", end=" ")
            client.set_model(server.get_global_model())
            client.set_optimizer(learning_rate)
            w, _, _ = client.train_local(epochs=local_epochs, verbose=verbose)
            val_loss, val_acc = client.validate()
            print(f"val_acc={val_acc:.2f}%")

            history[f"{client.client_id}_val_acc"].append(val_acc)
            cw_list.append((client.client_id, w, val_acc))

        # QPSO aggregation
        print("  QPSO aggregating …")
        agg = server.qpso_aggregate(cw_list)
        server.global_model.load_state_dict(agg)

        # track pbest / gbest
        for c in clients:
            history[f"{c.client_id}_pbest_score"].append(
                server.personal_best_scores[c.client_id])
        history["global_best_score"].append(server.global_best_score)

        # evaluate global
        g_acc, g_loss = server.evaluate_global_model(global_test_loader)
        dt = time.time() - t0

        history["round"].append(rnd)
        history["global_test_acc"].append(g_acc)
        history["global_test_loss"].append(g_loss)
        history["round_time"].append(dt)

        print(f"  Global  acc={g_acc:.2f}%  loss={g_loss:.4f}  "
              f"gbest={server.global_best_score:.2f}%  ({dt:.1f}s)")

        # best model
        if g_acc > best_acc:
            best_acc = g_acc
            torch.save(server.global_model.state_dict(),
                       f"{save_dir}/models/qpso_best.pth")
            print(f"  ✅ New best! {best_acc:.2f}%")

        # checkpoint
        if rnd % save_every == 0:
            torch.save(server.global_model.state_dict(),
                       f"{save_dir}/models/qpso_round_{rnd}.pth")

        # incremental CSV
        pd.DataFrame(history).to_csv(
            f"{save_dir}/results/qpso/metrics.csv", index=False)

    print("\n" + "=" * 80)
    print(f"  QPSO-FL DONE — best accuracy {best_acc:.2f}%  "
          f"gbest {server.global_best_score:.2f}%")
    print("=" * 80)
    return history
