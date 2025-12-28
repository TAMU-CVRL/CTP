import os
import torch
import clip
import argparse
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from torch.utils.data import DataLoader
from pathlib import Path

# Model and utility imports
from models.ctp import ctp
from models.clip.clip import tokenize
from utils.img_utils import image_transform
from utils.pc_utils import load_sparse_method
from utils.model_utils import get_clip_encoders, load_config, pc_backbone

# Dataset class imports aligned with CTPTrainer and provided scripts
from data.nuscenes_triplet import Triplet_Object_Nuscenes
from data.kitti_triplet import Triplet_Object_KITTI
from data.waymo_triplet import Triplet_Object_Waymo

class CTPEvaluator:
    """
    Evaluator class for CTP model with automatic dataset detection.
    Supports overrides for eval_path, loss_fn, and alpha via arguments.
    """
    def __init__(self, config_path, eval_path=None, loss_fn=None, alpha=None):
        self.cfg = load_config(config_path)
        self.device = self.cfg["Eval"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_template = self.cfg["Dataset"].get("prompt", "This is a ")
        
        # 1. Override configuration with command-line arguments if provided
        if eval_path:
            self.cfg["Eval"]["eval_data_path"] = eval_path
        if loss_fn:
            # Updating Model section as CTP model constructor reads from here
            self.cfg["Model"]["loss_fn"] = loss_fn
        if alpha is not None:
            self.cfg["Model"]["alpha"] = alpha

        # 2. Automatically detect dataset type from the JSONL path
        self.eval_path = self.cfg["Eval"]["eval_data_path"]
        self.dataset_type = self._detect_dataset_type(self.eval_path)
        print(f"[Info] Detected dataset type: {self.dataset_type}")
        print(f"[Info] Evaluation parameters: loss_fn={self.cfg['Model'].get('loss_fn')}, alpha={self.cfg['Model'].get('alpha')}")

        # 3. Initialize label mapping
        self._setup_label_mapping()
        
        # 4. Build model (Single-GPU)
        self.model = self._build_model()
        
        # 5. Load weights (Handles 'module.' prefix from DDP)
        self._load_checkpoint(self.cfg["Eval"]["checkpoint_path"])
        
        # 6. Build dataloader
        self.dataloader = self._build_dataloader()

    def _detect_dataset_type(self, path):
        """Detects the dataset type by searching for keywords in the file path."""
        path_lower = path.lower()
        if "kitti" in path_lower:
            return "kitti"
        elif "waymo" in path_lower:
            return "waymo"
        elif "nuscenes" in path_lower:
            return "nuscenes"
        else:
            print("[!] Warning: Dataset type not found in path. Defaulting to nuScenes.")
            return "nuscenes"

    def _setup_label_mapping(self):
        """Defines label mapping logic specific to each dataset's naming convention."""
        if self.dataset_type == "kitti":
            self.label_map = {
                'Car': 'car', 'Van': 'van', 'Truck': 'truck',
                'Pedestrian': 'pedestrian', 'Person_sitting': 'pedestrian',
                'Cyclist': 'pedestrian', 'Tram': None, 'Misc': None, 'DontCare': None
            }
        elif self.dataset_type == "waymo":
            self.label_map = {
                'Vehicle': 'car', 'Pedestrian': 'pedestrian',
                'Cyclist': 'pedestrian', 'Sign': 'sign'
            }
        else: # Default: nuscenes
            classes = [
                'car', 'truck', 'bus', 'pedestrian', 'bicycle', 
                'trailer', 'construction', 'motorcycle', 'barrier', 'trafficcone'
            ]
            self.label_map = {c: c for c in classes}

        self.merged_classes = sorted(set([v for v in self.label_map.values() if v]))
        self.all_prompts = [f"{self.prompt_template}{cls}" for cls in self.merged_classes]

    # def _handle_path(self, key):
    #     """Prioritizes local paths in TMPDIR if available for performance."""
    #     path = self.cfg["Dataset"].get(key, None)
    #     tmpdir = os.getenv("TMPDIR", None)
    #     if tmpdir and path:
    #         file_name = os.path.basename(path)
    #         local_path = os.path.join(tmpdir, file_name)
    #         if os.path.exists(local_path):
    #             return local_path
    #     return path

    def _build_model(self):
        """Instantiates the model components using the (potentially overridden) cfg."""
        pc_only = self.cfg["Model"]["pc_only"]
        clip_model_name = self.cfg["Model"]["clip_model"]
        pc_encoder_name = self.cfg["Model"]["point_model"].strip().lower()

        if pc_only:
            clip_model, _ = clip.load(clip_model_name, jit=False, device=self.device)
            clip_model.eval()
            for param in clip_model.parameters():
                param.requires_grad = False
            text_encoder, img_encoder = clip_model.encode_text, clip_model.encode_image
        else:
            text_encoder, img_encoder = get_clip_encoders(clip_model_name)
            text_encoder.to(self.device); img_encoder.to(self.device)

        pc_encoder = pc_backbone(pc_encoder_name, self.device)
        
        return ctp(
            text_encoder=text_encoder,
            image_encoder=img_encoder,
            lidar_encoder=pc_encoder,
            loss_fn=self.cfg["Model"]["loss_fn"],
            alpha=self.cfg["Model"]["alpha"]
        ).to(self.device)

    def _build_dataloader(self):
        """Initializes the correct Dataset class using the detected path."""
        sparse_method = self.cfg["Dataset"]["sparse_method"]
        sparse_to_dense = load_sparse_method(sparse_method)
        
        # image_tar = self._handle_path("image_tar_path")
        # lidar_tar = self._handle_path("lidar_tar_path")

        common_args = {
            "jsonl_file": self.eval_path,
            "image_transform": image_transform,
            "sparse_to_dense_fn": sparse_to_dense,
            "prompt": self.prompt_template
        }

        if self.dataset_type == "nuscenes":
            dataset = Triplet_Object_Nuscenes(**common_args)
        elif self.dataset_type == "kitti":
            dataset = Triplet_Object_KITTI(**common_args)
        elif self.dataset_type == "waymo":
            dataset = Triplet_Object_Waymo(**common_args)
        
        return DataLoader(
            dataset, 
            batch_size=self.cfg["Eval"]["batch_size"], 
            shuffle=False, 
            num_workers=self.cfg["Train"].get("num_workers", 4),
            pin_memory=True
        )

    def _load_checkpoint(self, path):
        """Loads state_dict and handles DDP prefixes."""
        print(f"[Info] Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        state_dict = ckpt['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)

    def run_evaluation(self):
        """Inference loop for computing the confusion matrix."""
        self.model.eval()
        confusion_matrix = torch.zeros(len(self.all_prompts), len(self.all_prompts))
        class_tokens = tokenize(self.all_prompts, truncate=True).to(self.device)

        with torch.no_grad():
            for sample in tqdm(self.dataloader, desc=f"Eval {self.dataset_type}"):
                raw_labels = sample["label"] 
                imgs = sample["image"].to(self.device)
                points = sample["lidar"].permute(0, 2, 1).to(self.device)

                text_feats, img_feats, pc_feats = self.model(class_tokens, imgs, points)
                _, sim = self.model.get_loss(text_feats, img_feats, pc_feats)
                preds = torch.argmax(sim, dim=1).cpu()

                for pred_idx, label in zip(preds, raw_labels):
                    clean_name = label.replace(self.prompt_template, "").strip()
                    mapped_gt = self.label_map.get(clean_name, None)
                    
                    if mapped_gt:
                        gt_prompt = f"{self.prompt_template}{mapped_gt}"
                        if gt_prompt in self.all_prompts:
                            gt_idx = self.all_prompts.index(gt_prompt)
                            confusion_matrix[gt_idx, pred_idx] += 1

        return self._summarize(confusion_matrix)

    def _summarize(self, confusion_matrix):
        """Calculates per-class and overall accuracy metrics."""
        results = {}
        total_hit, total_num = 0, 0
        
        for i, prompt in enumerate(self.all_prompts):
            row = confusion_matrix[i]
            count = row.sum().item()
            if count == 0: continue
            
            acc = row[i].item() / count
            cls_name = prompt.replace(self.prompt_template, "")
            
            probs = (row / count * 100)
            top_vals, top_idxs = torch.topk(probs, min(3, len(self.all_prompts)))
            top_info = ", ".join([f"{self.merged_classes[idx]}: {val:.1f}%" for val, idx in zip(top_vals, top_idxs)])
            
            results[cls_name] = {"acc": acc, "top3": top_info}
            total_hit += row[i].item()
            total_num += count

        overall_acc = total_hit / total_num if total_num > 0 else 0
        return results, overall_acc

    def log_results(self, results, overall_acc):
        """Logs metrics to console and saves to a text file."""
        save_dir = Path("results")
        save_dir.mkdir(exist_ok=True)
        log_path = save_dir / f"{self.cfg['Name']}_{self.dataset_type}_eval.txt"
        
        with open(log_path, "a") as f:
            f.write(f"\n--- {self.dataset_type.upper()} EVAL @ {datetime.now()} ---\n")
            f.write(f"Config: loss_fn={self.cfg['Model'].get('loss_fn')}, alpha={self.cfg['Model'].get('alpha')}\n")
            for cls, data in results.items():
                line = f"{cls.ljust(15)}: {data['acc']*100:6.2f}% | Top-3: {data['top3']}\n"
                print(line, end="")
                f.write(line)
            
            final_line = f"Overall Accuracy: {overall_acc*100:.2f}%\n"
            print("-" * 30 + f"\n{final_line}")
            f.write(final_line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CTP model with argument overrides.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    
    # New optional arguments to override config file
    parser.add_argument("--eval_path", type=str, required=True, help="Override eval_data_path in config")
    parser.add_argument("--loss_fn", type=str, default="l2_similarity_loss_completed", help="Override loss_fn in config")
    parser.add_argument("--alpha", type=float, default=None, help="Override alpha in config")
    
    args = parser.parse_args()

    # Pass arguments to constructor for overriding
    evaluator = CTPEvaluator(
        config_path=args.config, 
        eval_path=args.eval_path, 
        loss_fn=args.loss_fn, 
        alpha=args.alpha
    )
    
    res, acc = evaluator.run_evaluation()
    evaluator.log_results(res, acc)
