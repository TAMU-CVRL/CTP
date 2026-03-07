import argparse
from CTPEvaluator import CTPEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CTP model with argument overrides.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--eval_path", type=str, default="dataset/nuscenes_triplets/nuscenes_triplet_val.jsonl", help="Override eval_data_path in config")
    parser.add_argument("--loss_fn", type=str, default="cosine_matrix_loss_eval", help="Override loss_fn in config")
    parser.add_argument("--tau", type=float, default=0.5, help="tau = 0: Text - Point; tau = 1: Text - Image; 0.5: Text - Image + Point")
    args = parser.parse_args()

    evaluator = CTPEvaluator(
        config_path=args.config, 
        eval_path=args.eval_path, 
        loss_fn=args.loss_fn, 
        tau=args.tau
    )
    
    res, acc = evaluator.run_evaluation()
    evaluator.log_results(res, acc)
