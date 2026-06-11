
import torch
import argparse
import os
import json

from src.model import PECL
from src.lightgcn import create_adjacency_matrix
from src.utils import evaluate_model
from data_loader import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate PECL checkpoint")
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    data_root = args.data_root or os.path.join(os.path.dirname(__file__), "..", "data")
    print(f"Loading {args.dataset} ...")
    train_interactions, test_interactions, timestamps, n_users, n_items = load_dataset(
        args.dataset, data_root,
    )

    adj_matrix = create_adjacency_matrix(n_users, n_items, train_interactions).to(device)


    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    m_args = ckpt.get("args", {})

    model = PECL(
        n_users, n_items,
        embed_dim=m_args.get("embed_dim", 64),
        n_layers=m_args.get("n_layers", 3),
        alpha=m_args.get("alpha", 2),
        beta=m_args.get("beta", 4),
        tau=m_args.get("tau", 0.05),
    )
    model.set_path_sampler(train_interactions, timestamps)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()


    print("Evaluating ...")
    with torch.no_grad():
        u_emb, i_emb = model(adj_matrix)
        metrics = evaluate_model(
            model, train_interactions, test_interactions, u_emb, i_emb,
        )

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for m, v in metrics.items():
        print(f"  {m}: {v:.4f}")
    print("=" * 50)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"{args.dataset}_results.json")
    with open(out, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
