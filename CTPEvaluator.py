import torch
import clip
import argparse
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from pathlib import Path

from models.ctp import ctp
from models.clip.clip import tokenize
from utils.img_utils import image_transform
from utils.pc_utils import load_sparse_method
from utils.model_utils import get_clip_encoders, load_config, pc_backbone

from data.nuscenes_triplet import Triplet_Object_Nuscenes
from data.kitti_triplet import Triplet_Object_KITTI
from data.waymo_triplet import Triplet_Object_Waymo

class CTPEvaluator:
    def __init__(self, config_path, eval_path=None, loss_fn=None, alpha=None):
        self.cfg = load_config(config_path)
        self.device = self.cfg["Eval"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_template = self.cfg["Dataset"].get("prompt", "This is a ")
        
        # Override configuration with command-line arguments if provided
        if eval_path:
            self.cfg["Eval"]["eval_data_path"] = eval_path
        if loss_fn:
            # Updating Model section as CTP model constructor reads from here
            self.cfg["Model"]["loss_fn"] = loss_fn
        if alpha is not None:
            self.cfg["Model"]["alpha"] = alpha

        # Automatically detect dataset type from the JSONL path
        self.eval_path = self.cfg["Eval"]["eval_data_path"]
        self.dataset_type = self._detect_dataset_type(self.eval_path)
        print(f"[Info] Detected dataset type: {self.dataset_type}")
        print(f"[Info] Evaluation parameters: loss_fn={self.cfg['Model'].get('loss_fn')}, alpha={self.cfg['Model'].get('alpha')}")

        self._setup_label_mapping()
        self.model = self._build_model()
        self._load_checkpoint(self.cfg["Eval"]["checkpoint_path"])
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

    def _extract_triplet_features(self, target_label, max_samples=50):
        """Extract features, with tqdm bound to max_samples"""
        self.model.eval()
        t_list, i_list, l_list = [], [], []
        collected = 0
        target_prompt = f"{self.prompt_template}{target_label}"
        pbar = tqdm(total=max_samples, desc=f"Extracting {target_label}")

        with torch.no_grad():
            for batch in self.dataloader:
                labels = batch["label"] 
                imgs = batch["image"].to(self.device)
                points = batch["lidar"].permute(0, 2, 1).to(self.device)
                # Prioritize rich captions; if none, use fixed template
                captions = batch.get("caption", [target_prompt] * len(labels))

                for i in range(len(labels)):
                    clean_name = labels[i].replace(self.prompt_template, "").strip()
                    mapped_name = self.label_map.get(clean_name, None)
                    
                    if mapped_name != target_label:
                        continue

                    txt_token = tokenize(captions[i], truncate=True).to(self.device)
                    t_f, i_f, l_f = self.model(txt_token, imgs[i:i+1], points[i:i+1])
                    
                    t_list.append(t_f.cpu().numpy())
                    i_list.append(i_f.cpu().numpy())
                    l_list.append(l_f.cpu().numpy())

                    collected += 1
                    pbar.update(1)
                    if collected >= max_samples: break
                if collected >= max_samples: break
        
        pbar.close()
        return {"text": np.vstack(t_list), "image": np.vstack(i_list), "lidar": np.vstack(l_list)}

    def _draw_comparison_plot(self, emb, n, reduction_method, target_label, after_ckpt_path, before_ckpt_path=None, save_name=None):
        plt.rcParams.update({'font.size': 14, 'legend.fontsize': 12})
        
        plt.figure(figsize=(10, 10))

        # Before state (first 3*n rows)
        bt = emb[:n]                  # before text
        bi = emb[n:2*n]               # before image
        bl = emb[2*n:3*n]             # before lidar
        
        # After state (last 3*n rows)
        at = emb[3*n:4*n]             # after text
        ai = emb[4*n:5*n]             # after image
        al = emb[5*n:6*n]             # after lidar

        # Plot BEFORE state (using light colors, no border)
        plt.scatter(bt[:, 0], bt[:, 1], s=60, c="#a6cee3", marker='s', alpha=0.6, label="Text (Before)")
        plt.scatter(bi[:, 0], bi[:, 1], s=60, c="#fb9a99", marker='^', alpha=0.6, label="Image (Before)")
        plt.scatter(bl[:, 0], bl[:, 1], s=60, c="#b2df8a", marker='o', alpha=0.6, label="LiDAR (Before)")

        # Plot AFTER state (using dark colors with black border)
        plt.scatter(at[:, 0], at[:, 1], s=70, c="#1f78b4", marker='s', edgecolors='black', label="Text (After)")
        plt.scatter(ai[:, 0], ai[:, 1], s=70, c="#e31a1c", marker='^', edgecolors='black', label="Image (After)")
        plt.scatter(al[:, 0], al[:, 1], s=70, c="#33a02c", marker='o', edgecolors='black', label="LiDAR (After)")

        # Chart auxiliary information
        dataset_name = Path(self.eval_path).stem
        plt.title(f"{reduction_method.upper()} Alignment Comparison\nDataset: {dataset_name}\nClass: {target_label}\nckpt: {after_ckpt_path}", pad=20)
        plt.legend(loc="best", frameon=True)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Automatic generation and save path
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            save_name = f"{self.cfg['Name']}_{reduction_method}_{target_label}_{timestamp}.png"
        
        save_dir = Path("results") / "alignment"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / save_name
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"[Success] {reduction_method.upper()} plot saved to: {save_path}")

    def plot_embedding_comparison(self, target_label, after_ckpt_path, before_ckpt_path=None, 
                                  reduction_method="umap", max_samples=50, **kwargs):
        """Integrates feature comparison plot generation and multi-modal Acc calculation"""
        
        # 1. Extract 'Before' state
        if before_ckpt_path:
            self._load_checkpoint(before_ckpt_path)
        else:
            print("\n[Info] No 'before_ckpt' provided. Using random initial state.")
            self.model = self._build_model() # Randomization
        
        before_feats = self._extract_triplet_features(target_label, max_samples)

        # 2. Extract 'After' state and calculate Acc
        self._load_checkpoint(after_ckpt_path)
        after_feats = self._extract_triplet_features(target_label, max_samples)

        # 3. Dimensionality reduction and plotting
        all_data = np.concatenate([
            before_feats["text"], before_feats["image"], before_feats["lidar"],
            after_feats["text"], after_feats["image"], after_feats["lidar"]
        ], axis=0)

        if reduction_method.lower() == "umap":
            reducer = umap.UMAP(n_neighbors=kwargs.get("n_neighbors", 15), random_state=831)
        else:
            reducer = TSNE(n_components=2, perplexity=kwargs.get("perplexity", 30.0), random_state=831)

        emb = reducer.fit_transform(all_data)
        
        self._draw_comparison_plot(
            emb,
            before_feats["text"].shape[0], 
            reduction_method, 
            target_label,
            after_ckpt_path
        )

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
        save_dir = Path("results") / "classification"
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / f"{self.cfg['Name']}_{self.dataset_type}_eval.txt"
        dataset_name = Path(self.eval_path).stem

        with open(log_path, "a") as f:
            f.write(f"\n------ {self.dataset_type.upper()} EVAL @ {datetime.now()} ------\n")
            f.write(f"Config: loss_fn={self.cfg['Model'].get('loss_fn')}\nalpha: {self.cfg['Model'].get('alpha')}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write("-" * 30 + "\n")
            for cls, data in results.items():
                line = f"{cls.ljust(15)}: {data['acc']*100:6.2f}% | Top-3: {data['top3']}\n"
                print(line, end="")
                f.write(line)
            f.write("-" * 30 + "\n")
            final_line = f"Overall Accuracy: {overall_acc*100:.2f}%\n"
            print("-" * 30 + f"\n{final_line}")
            f.write(final_line)

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
