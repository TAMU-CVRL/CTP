import argparse
from CTPEvaluator import CTPEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize embedding alignment using UMAP or t-SNE.")
    
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--eval_path", type=str, required=True, help="JSONL evaluation data path")
    parser.add_argument("--loss_fn", type=str, default="l2_similarity_loss_completed")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--before_ckpt", type=str, default=None, help="Initial weights (Optional)")
    parser.add_argument("--after_ckpt", type=str, required=True, help="Trained weights (Required)")
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne"], help="Reduction method")
    parser.add_argument("--label", type=str, default="car", choices=["car", "truck", "pedestrian"], help="Class to visualize (e.g., car, truck, pedestrian)")
    parser.add_argument("--max_samples", type=int, default=200, help="Max samples per modality")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors")
    
    args = parser.parse_args()

    evaluator = CTPEvaluator(
        config_path=args.config, 
        eval_path=args.eval_path, 
        loss_fn=args.loss_fn, 
        alpha=args.alpha
    )
    
    evaluator.plot_embedding_comparison(
        target_label=args.label,
        before_ckpt_path=args.before_ckpt,
        after_ckpt_path=args.after_ckpt,
        reduction_method=args.method,
        max_samples=args.max_samples,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors
    )
