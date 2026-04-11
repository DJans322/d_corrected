import os
import json
import time
from collections import defaultdict


class TrainingLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.log_file = os.path.join(output_dir, "train_log.jsonl")
        self.txt_file = os.path.join(output_dir, "train_log.txt")

        self.start_time = time.time()

    def log(self, data: dict):
        """
        写入一条日志（JSON + 可读文本）
        """
        data["time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # JSON（推荐后期画图用）
        with open(self.log_file, "a") as f:
            f.write(json.dumps(data) + "\n")

        # 文本（方便看）
        with open(self.txt_file, "a") as f:
            f.write(self.format_log(data) + "\n")

    def format_log(self, data):
        msg = []
        for k, v in data.items():
            if isinstance(v, float):
                msg.append(f"{k}: {v:.4f}")
            else:
                msg.append(f"{k}: {v}")
        return " | ".join(msg)

    def log_epoch(self, epoch, loss=None, lr=None):
        self.log({
            "type": "epoch",
            "epoch": epoch,
            "loss": loss,
            "lr": lr
        })

    def log_cluster(self, epoch, pseudo_labels):
        import numpy as np

        labels = np.array(pseudo_labels)

        valid = labels[labels != -1]
        num_clusters = len(set(valid))
        num_noise = np.sum(labels == -1)

        cluster_sizes = []
        for l in set(valid):
            cluster_sizes.append(np.sum(valid == l))

        self.log({
            "type": "cluster",
            "epoch": epoch,
            "num_clusters": num_clusters,
            "num_samples": len(labels),
            "num_noise": int(num_noise),
            "noise_ratio": float(num_noise / len(labels)),
            "avg_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0,
            "max_cluster_size": int(np.max(cluster_sizes)) if cluster_sizes else 0,
            "min_cluster_size": int(np.min(cluster_sizes)) if cluster_sizes else 0,
        })

    def log_split_merge(self, epoch, split_cnt, merge_cnt, avg_am=None, avg_bm=None):
        self.log({
            "type": "split_merge",
            "epoch": epoch,
            "split_clusters": split_cnt,
            "merge_clusters": merge_cnt,
            "avg_a_m": avg_am,
            "avg_b_m": avg_bm
        })

    def log_eval(self, epoch, mAP=None, top1=None):
        self.log({
            "type": "eval",
            "epoch": epoch,
            "mAP": mAP,
            "top1": top1
        })