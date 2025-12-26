import CTPTrainer
import argparse
from utils.model_utils import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/ctp_default.yaml",
                        help="Path to the config file.")
    args = parser.parse_args()
    
    config = load_config(args.config_path)
    trainer = CTPTrainer.CTPTrainer(config)
    trainer.train()
