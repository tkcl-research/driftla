
import torch
import torch.optim as optim
import numpy as np
import argparse
import os
import json
from collections import defaultdict
from tqdm import tqdm

from src.model import PECL
from src.lightgcn import create_adjacency_matrix
from src.utils import set_seed, evaluate_model
from data_loader import load_dataset, create_negative_samples


def train_epoch(model, train_interactions, user_pos_items, adj_matrix, n_items,
                batch_size, lambda1, lambda2, lambda3,
                timestamps_dict, device, optimizer):
    model.train()
    total_loss = 0.0
    comp = {"bpr": 0.0, "intra": 0.0, "inter": 0.0, "l2": 0.0}

    indices = np.random.permutation(len(train_interactions))
    n_batches = (len(train_interactions) + batch_size - 1) // batch_size

    for b in tqdm(range(n_batches), desc="Training", leave=False):
        lo = b * batch_size
        hi = min(lo + batch_size, len(train_interactions))
        batch_idx = indices[lo:hi]

        users_np = np.array([train_interactions[j][0] for j in batch_idx])
        items_np = np.array([train_interactions[j][1] for j in batch_idx])

        users = torch.tensor(users_np, dtype=torch.long, device=device)
        pos_items = torch.tensor(items_np, dtype=torch.long, device=device)
        neg_items_np = create_negative_samples(users_np, user_pos_items, n_items)
        neg_items = torch.tensor(neg_items_np, dtype=torch.long, device=device)

        optimizer.zero_grad()
        loss, c = model.compute_total_loss(
            users, pos_items, neg_items, adj_matrix,
            lambda1, lambda2, lambda3, timestamps_dict,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k in comp:
            comp[k] += c[k]

    for k in comp:
        comp[k] /= n_batches
    return total_loss / n_batches, comp


def main():
    parser = argparse.ArgumentParser(description="Train PECL on ML-1M")
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root dir containing dataset folder (default: ../data)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--alpha", type=int, default=2)
    parser.add_argument("--beta", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument("--lambda3", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--eval_every", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    data_root = args.data_root or os.path.join(os.path.dirname(__file__), "..", "data")
    print(f"Loading {args.dataset} dataset from {data_root} ...")
    train_interactions, test_interactions, timestamps, n_users, n_items = load_dataset(
        args.dataset, data_root,
    )
    print(f"  Users: {n_users},  Items: {n_items}")
    print(f"  Train: {len(train_interactions)},  Test: {len(test_interactions)}")


    user_pos = defaultdict(set)
    for u, i in train_interactions:
        user_pos[u].add(i)


    print("Creating adjacency matrix ...")
    adj_matrix = create_adjacency_matrix(n_users, n_items, train_interactions).to(device)


    model = PECL(
        n_users, n_items,
        embed_dim=args.embed_dim, n_layers=args.n_layers,
        alpha=args.alpha, beta=args.beta, tau=args.tau,
    )
    model.set_path_sampler(train_interactions, timestamps)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.save_dir, exist_ok=True)

    best_ndcg = 0.0

    print("\nStarting training ...\n")
    for epoch in range(1, args.epochs + 1):
        avg_loss, comp = train_epoch(
            model, train_interactions, user_pos, adj_matrix, n_items,
            args.batch_size, args.lambda1, args.lambda2, args.lambda3,
            timestamps, device, optimizer,
        )

        do_eval = (epoch % args.eval_every == 0) or (epoch == 1) or (epoch == args.epochs)
        if do_eval:
            model.eval()
            with torch.no_grad():
                u_emb, i_emb = model(adj_matrix)
                metrics = evaluate_model(
                    model, train_interactions, test_interactions, u_emb, i_emb,
                )
            model.train()

            print(f"Epoch {epoch}/{args.epochs}  Loss: {avg_loss:.4f}  "
                  f"BPR={comp['bpr']:.4f}  Intra={comp['intra']:.4f}  "
                  f"Inter={comp['inter']:.4f}  L2={comp['l2']:.6f}")
            for m, v in metrics.items():
                print(f"  {m}: {v:.4f}")

            ndcg10 = metrics.get("NDCG@10", 0.0)
            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                }
                path = os.path.join(args.save_dir, f"pecl_{args.dataset}_best.pth")
                torch.save(ckpt, path)
                print(f"  ** Saved best model (NDCG@10={best_ndcg:.4f}) **")
            print()


    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)


    best_path = os.path.join(args.save_dir, f"pecl_{args.dataset}_best.pth")
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        final_metrics = best_ckpt["metrics"]
    else:
        final_metrics = metrics

    results_file = os.path.join(results_dir, f"{args.dataset}_results.json")
    with open(results_file, "w") as f:
        json.dump({k: float(v) for k, v in final_metrics.items()}, f, indent=2)

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for m, v in final_metrics.items():
        print(f"  {m}: {v:.4f}")
    print(f"Results saved to {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
