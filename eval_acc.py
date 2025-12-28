import argparse
from CTPEvaluator import CTPEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CTP model with argument overrides.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--eval_path", type=str, required=True, help="Override eval_data_path in config")
    parser.add_argument("--loss_fn", type=str, default="l2_similarity_loss_completed", help="Override loss_fn in config")
    parser.add_argument("--alpha", type=float, default=None, help="Override alpha in config")
    
    args = parser.parse_args()

    evaluator = CTPEvaluator(
        config_path=args.config, 
        eval_path=args.eval_path, 
        loss_fn=args.loss_fn, 
        alpha=args.alpha
    )
    
    res, acc = evaluator.run_evaluation()
    evaluator.log_results(res, acc)
