"""
train_late_fusion.py

Training script cho LateFusionModel: ETH_GBert (text+graph) + GRU (time-series).

Cách chạy:
    python train_late_fusion.py --ds Dataset --dim 16 --fusion attention
    python train_late_fusion.py --ds Dataset --dim 16 --fusion concat --freeze_ethgbert
    python train_late_fusion.py --ds Dataset --bert_dir ./mlm_output/

Yêu cầu dữ liệu:
    1) ETH_GBert preprocessed: data/preprocessed/Dataset/
       - data_Dataset.labels, .train_y, .valid_y, .test_y, ...
       - data_Dataset.shuffled_clean_docs
       - data_Dataset.address_to_index
       - norm_adj_coo.npz
    2) GRU time-series (cùng thứ tự account với ETH_GBert):
       - data/preprocessed/Dataset/X_timeseries_train.npy
       - data/preprocessed/Dataset/X_timeseries_valid.npy
       - data/preprocessed/Dataset/X_timeseries_test.npy
       Hoặc 1 file chung:
       - data/preprocessed/Dataset/X_timeseries.npy  (sẽ tự split theo cùng size)
"""

import argparse
import gc
import os
import pickle as pkl
import random
import time
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

from env_config import env_config
from ETH_GBert import ETH_GBertModel
from LateFusion import (
    AttentionFusion,
    BiGRUBranch,
    CNNGRUBranch,
    ConcatFusion,
    ETHGBertBranch,
    GRUAttentionBranch,
    GRUBranch,
    LateFusionModel,
    TCNBranch,
)
from utils import (
    InputExample,
    example2feature,
    get_class_count_and_weight,
    sparse_scipy2torch,
)


# ============================================================
# 0. GLOBAL SETUP
# ============================================================
random.seed(env_config.GLOBAL_SEED)
np.random.seed(env_config.GLOBAL_SEED)
torch.manual_seed(env_config.GLOBAL_SEED)

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    torch.cuda.manual_seed_all(env_config.GLOBAL_SEED)

DEVICE = torch.device("cuda:0" if CUDA_AVAILABLE else "cpu")
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 1. ARGUMENT PARSING
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train LateFusion: ETH_GBert + GRU")

    # Data
    parser.add_argument("--ds", type=str, default="Dataset")
    parser.add_argument("--timeseries_dir", type=str, default="",
                        help="Thư mục chứa X_timeseries*.npy. Mặc định = data/preprocessed/<ds>/")

    # Model
    parser.add_argument("--bert_dir", type=str, default="",
                        help="Đường dẫn tới domain-adapted BERT checkpoint (MLM output)")
    parser.add_argument("--dim", type=int, default=16,
                        help="GCN embedding dim cho ETH_GBert")
    parser.add_argument("--fusion", type=str, default="attention",
                        choices=["attention", "concat"],
                        help="Phương pháp fusion: attention hoặc concat")
    parser.add_argument("--common_dim", type=int, default=256,
                        help="Chiều fused representation")

    # GRU config
    parser.add_argument("--branch", type=str, default="gru",
                        choices=["gru", "gru_attention", "bigru", "tcn", "cnn_gru"],
                        help="Loại branch time-series: gru, gru_attention, bigru, tcn, cnn_gru")
    parser.add_argument("--gru_input_size", type=int, default=10,
                        help="Số features mỗi timestep")
    parser.add_argument("--gru_hidden_size", type=int, default=64)
    parser.add_argument("--gru_num_layers", type=int, default=2)

    # Training
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)

    # Freeze
    parser.add_argument("--freeze_ethgbert", action="store_true",
                        help="Đóng băng ETH_GBert, chỉ train fusion + classifier")
    parser.add_argument("--freeze_gru", action="store_true",
                        help="Đóng băng GRU")

    # Misc
    parser.add_argument("--load", type=int, default=0, help="1=resume from checkpoint")
    parser.add_argument("--sw", type=int, default=0)
    parser.add_argument("--validate_program", action="store_true",
                        help="Chỉ chạy 1 epoch với 1 sample để test pipeline")
    parser.add_argument("--output_dir", type=str, default="./output_fusion/")

    return parser.parse_args()


# ============================================================
# 2. CONFIG BUILDER
# ============================================================
def build_config(args):
    cfg = {
        "dataset": args.ds,
        "branch_type": args.branch,
        "fusion_type": args.fusion,
        "common_dim": args.common_dim,
        "gcn_embedding_dim": args.dim,
        "gru_input_size": args.gru_input_size,
        "gru_hidden_size": args.gru_hidden_size,
        "gru_num_layers": args.gru_num_layers,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.l2,
        "dropout_rate": args.dropout,
        "warmup_proportion": args.warmup,
        "gradient_accumulation_steps": args.grad_accum,
        "total_train_epochs": args.epochs,
        "freeze_ethgbert": args.freeze_ethgbert,
        "freeze_gru": args.freeze_gru,
        "do_lower_case": True,
        "do_softmax_before_mse": True,
        "loss_criterion": "cle",
        "output_dir": args.output_dir,
        "perform_metrics_str": ["weighted avg", "f1-score"],
        "bert_model_scale": "bert-base-uncased",
    }

    # BERT path
    if args.bert_dir:
        cfg["bert_model_scale"] = args.bert_dir
    elif env_config.TRANSFORMERS_OFFLINE == 1:
        cfg["bert_model_scale"] = os.path.join(
            env_config.HUGGING_LOCAL_MODEL_FILES_PATH,
            f"hf-maintainers_{cfg['bert_model_scale']}",
        )

    cfg["max_seq_length"] = 200 + cfg["gcn_embedding_dim"]
    cfg["data_dir"] = f"data/preprocessed/{args.ds}"
    cfg["timeseries_dir"] = args.timeseries_dir or cfg["data_dir"]

    if args.validate_program:
        cfg["total_train_epochs"] = 1

    cfg["model_file"] = (
        f"LateFusion_{cfg['branch_type']}_{cfg['fusion_type']}"
        f"_{cfg['gcn_embedding_dim']}_{cfg['gru_hidden_size']}_{args.ds}.pt"
    )

    os.makedirs(cfg["output_dir"], exist_ok=True)
    return cfg


def print_config(cfg):
    print("\n" + "=" * 60)
    print("  LATE FUSION TRAINING")
    print("=" * 60)
    print(f"  Dataset:          {cfg['dataset']}")
    print(f"  Branch type:      {cfg['branch_type']}")
    print(f"  Fusion type:      {cfg['fusion_type']}")
    print(f"  Common dim:       {cfg['common_dim']}")
    print(f"  GCN embed dim:    {cfg['gcn_embedding_dim']}")
    print(f"  GRU hidden:       {cfg['gru_hidden_size']}")
    print(f"  Batch size:       {cfg['batch_size']}")
    print(f"  Learning rate:    {cfg['learning_rate']}")
    print(f"  Weight decay:     {cfg['weight_decay']}")
    print(f"  Dropout:          {cfg['dropout_rate']}")
    print(f"  Epochs:           {cfg['total_train_epochs']}")
    print(f"  Freeze ETH_GBert: {cfg['freeze_ethgbert']}")
    print(f"  Freeze GRU:       {cfg['freeze_gru']}")
    print(f"  BERT source:      {cfg['bert_model_scale']}")
    print(f"  Loss:             {cfg['loss_criterion']}")
    print(f"  Device:           {DEVICE}")
    print(f"  Model file:       {cfg['model_file']}")
    print("=" * 60 + "\n")


# ============================================================
# 3. DATA LOADING
# ============================================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f, encoding="latin1")


def load_ethgbert_data(cfg):
    """Load toàn bộ preprocessed data cho ETH_GBert branch."""
    data_dir = cfg["data_dir"]
    ds = cfg["dataset"]

    print("[ETH_GBert data] Loading...")
    names = [
        "labels", "train_y", "train_y_prob",
        "valid_y", "valid_y_prob",
        "test_y", "test_y_prob",
        "shuffled_clean_docs", "address_to_index",
    ]
    objects = {}
    for name in names:
        path = os.path.join(data_dir, f"data_{ds}.{name}")
        objects[name] = load_pickle(path)
        print(f"  Loaded: {name}")

    # GCN adjacency
    npz = np.load(os.path.join(data_dir, "norm_adj_coo.npz"))
    a_hat = sp.coo_matrix(
        (npz["data"], (npz["row"], npz["col"])),
        shape=tuple(npz["shape"]),
    )
    gcn_adj_list = [sparse_scipy2torch(a_hat).to(DEVICE)]
    print(f"  Loaded: norm_adj_coo.npz (shape={a_hat.shape}, nnz={a_hat.nnz})")

    return objects, gcn_adj_list


def load_timeseries_data(cfg, train_size, valid_size, test_size):
    """
    Load time-series data cho GRU branch.

    Hỗ trợ 2 cách:
      A) 3 file riêng: X_timeseries_train.npy, X_timeseries_valid.npy, X_timeseries_test.npy
      B) 1 file chung:  X_timeseries.npy → split theo train_size/valid_size/test_size
    """
    ts_dir = cfg["timeseries_dir"]
    print("[GRU data] Loading time-series...")

    train_path = os.path.join(ts_dir, "X_timeseries_train.npy")
    valid_path = os.path.join(ts_dir, "X_timeseries_valid.npy")
    test_path = os.path.join(ts_dir, "X_timeseries_test.npy")

    if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path):
        # Cách A: 3 file riêng
        X_train = np.load(train_path)
        X_valid = np.load(valid_path)
        X_test = np.load(test_path)
        print(f"  Loaded 3 separate files")
    else:
        # Cách B: 1 file chung, split theo cùng thứ tự với ETH_GBert
        all_path = os.path.join(ts_dir, "X_timeseries.npy")
        if not os.path.exists(all_path):
            raise FileNotFoundError(
                f"Không tìm thấy time-series data.\n"
                f"  Cần 1 trong 2:\n"
                f"    A) {train_path}, {valid_path}, {test_path}\n"
                f"    B) {all_path}\n"
                f"  Đảm bảo thứ tự account khớp với ETH_GBert preprocessed data."
            )
        X_all = np.load(all_path)
        total = train_size + valid_size + test_size

        if len(X_all) != total:
            raise ValueError(
                f"X_timeseries.npy có {len(X_all)} samples, "
                f"nhưng ETH_GBert data có {total} samples "
                f"(train={train_size}, valid={valid_size}, test={test_size}).\n"
                f"Hai bên phải cùng số lượng và cùng thứ tự account."
            )

        X_train = X_all[:train_size]
        X_valid = X_all[train_size: train_size + valid_size]
        X_test = X_all[train_size + valid_size:]
        print(f"  Loaded {all_path} and split by ETH_GBert sizes")

    print(f"  Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
    return X_train, X_valid, X_test


def build_examples(objects):
    """Tạo InputExample list cho ETH_GBert branch (giống train.py gốc)."""
    label2idx = objects["labels"][0]
    idx2label = objects["labels"][1]
    train_y = objects["train_y"]
    valid_y = objects["valid_y"]
    test_y = objects["test_y"]
    y = np.hstack((train_y, valid_y, test_y))
    y_prob = np.vstack((objects["train_y_prob"], objects["valid_y_prob"], objects["test_y_prob"]))
    docs = objects["shuffled_clean_docs"]

    examples = []
    for i, text in enumerate(docs):
        examples.append(InputExample(i, text.strip(), confidence=y_prob[i], label=y[i]))

    train_size = len(train_y)
    valid_size = len(valid_y)
    test_size = len(test_y)

    train_examples = examples[:train_size]
    valid_examples = examples[train_size: train_size + valid_size]
    test_examples = examples[train_size + valid_size:]

    return {
        "label2idx": label2idx,
        "idx2label": idx2label,
        "train_examples": train_examples,
        "valid_examples": valid_examples,
        "test_examples": test_examples,
        "num_classes": len(label2idx),
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size,
        "train_y": train_y,
    }


# ============================================================
# 4. FUSION DATASET — kết hợp ETH_GBert + GRU data
# ============================================================
class FusionDataset(Dataset):
    """
    Dataset kết hợp:
      - ETH_GBert features (text tokenized + GCN vocab ids)
      - GRU features (time-series tensor)

    Mỗi sample trả về tuple:
      (input_ids, input_mask, segment_ids, confidence, label_id, gcn_vocab_ids, timeseries)
    """

    def __init__(self, examples, X_timeseries, tokenizer, gcn_vocab_map,
                 max_seq_len, gcn_embedding_dim):
        assert len(examples) == len(X_timeseries), (
            f"Số lượng text examples ({len(examples)}) != "
            f"số lượng timeseries samples ({len(X_timeseries)}). "
            f"Hai bên phải khớp 1-1 theo account."
        )
        self.examples = examples
        self.X_timeseries = X_timeseries
        self.tokenizer = tokenizer
        self.gcn_vocab_map = gcn_vocab_map
        self.max_seq_len = max_seq_len
        self.gcn_embedding_dim = gcn_embedding_dim

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat = example2feature(
            self.examples[idx],
            self.tokenizer,
            self.gcn_vocab_map,
            self.max_seq_len,
            self.gcn_embedding_dim,
        )
        ts = self.X_timeseries[idx]  # (seq_len, n_features)
        return (
            feat.input_ids,
            feat.input_mask,
            feat.segment_ids,
            feat.confidence,
            feat.label_id,
            feat.gcn_vocab_ids,
            ts,
        )

    def pad(self, batch):
        """
        Custom collate: pad ETH_GBert sequences + stack GRU tensors.
        """
        gcn_vocab_size = len(self.gcn_vocab_map)
        seq_lens = [len(sample[0]) for sample in batch]
        max_len = np.max(seq_lens)

        def collect(i):
            return [sample[i] for sample in batch]

        def pad_1d(i, target_len):
            return [
                sample[i] + [0] * (target_len - len(sample[i]))
                for sample in batch
            ]

        def pad_gcn_ids(i, target_len):
            return [
                [-1] + sample[i] + [-1] * (target_len - len(sample[i]) - 1)
                for sample in batch
            ]

        # ETH_GBert tensors
        batch_input_ids = torch.tensor(pad_1d(0, max_len), dtype=torch.long)
        batch_input_mask = torch.tensor(pad_1d(1, max_len), dtype=torch.long)
        batch_segment_ids = torch.tensor(pad_1d(2, max_len), dtype=torch.long)
        batch_confidences = torch.tensor(collect(3), dtype=torch.float)
        batch_label_ids = torch.tensor(collect(4), dtype=torch.long)

        batch_gcn_vocab_ids_padded = np.array(pad_gcn_ids(5, max_len)).reshape(-1)
        batch_gcn_swop_eye = torch.eye(gcn_vocab_size + 1)[
            batch_gcn_vocab_ids_padded
        ][:, :-1]
        batch_gcn_swop_eye = batch_gcn_swop_eye.view(
            len(batch), -1, gcn_vocab_size
        ).transpose(1, 2)

        # GRU tensor
        batch_timeseries = torch.tensor(
            np.array(collect(6)), dtype=torch.float32
        )

        return (
            batch_input_ids,
            batch_input_mask,
            batch_segment_ids,
            batch_confidences,
            batch_label_ids,
            batch_gcn_swop_eye,
            batch_timeseries,
        )


def build_fusion_dataloader(examples, X_timeseries, tokenizer, address_to_index,
                            max_seq_length, gcn_embedding_dim, batch_size, shuffle=False):
    dataset = FusionDataset(
        examples=examples,
        X_timeseries=X_timeseries,
        tokenizer=tokenizer,
        gcn_vocab_map=address_to_index,
        max_seq_len=max_seq_length,
        gcn_embedding_dim=gcn_embedding_dim,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        collate_fn=dataset.pad,
    )


# ============================================================
# 5. MODEL CONSTRUCTION
# ============================================================
def build_model(cfg, label2idx, gcn_vocab_size, gcn_adj_list):
    """Tạo LateFusionModel từ config."""
    print("\n[Model] Building LateFusionModel...")

    # --- ETH_GBert branch ---
    print(f"  Loading ETH_GBert from: {cfg['bert_model_scale']}")
    ethgbert_model = ETH_GBertModel.from_pretrained(
        cfg["bert_model_scale"],
        gcn_adj_dim=gcn_vocab_size,
        gcn_adj_num=len(gcn_adj_list),
        gcn_embedding_dim=cfg["gcn_embedding_dim"],
        num_labels=len(label2idx),
    )
    ethgbert_branch = ETHGBertBranch(ethgbert_model)
    print(f"  ETH_GBert branch output: {ethgbert_branch.output_dim}-dim")

    # --- Time-series branch (chọn theo --branch) ---
    branch_type = cfg["branch_type"]
    branch_kwargs = dict(
        input_size=cfg["gru_input_size"],
        hidden_size=cfg["gru_hidden_size"],
        num_layers=cfg["gru_num_layers"],
        dropout=cfg["dropout_rate"],
    )

    if branch_type == "gru":
        ts_branch = GRUBranch(**branch_kwargs)
    elif branch_type == "gru_attention":
        ts_branch = GRUAttentionBranch(**branch_kwargs)
    elif branch_type == "bigru":
        ts_branch = BiGRUBranch(**branch_kwargs)
    elif branch_type == "tcn":
        ts_branch = TCNBranch(
            input_size=cfg["gru_input_size"],
            hidden_size=cfg["gru_hidden_size"],
            num_layers=cfg["gru_num_layers"],
            dropout=cfg["dropout_rate"],
        )
    elif branch_type == "cnn_gru":
        ts_branch = CNNGRUBranch(
            input_size=cfg["gru_input_size"],
            hidden_size=cfg["gru_hidden_size"],
            num_layers=cfg["gru_num_layers"],
            dropout=cfg["dropout_rate"],
        )
    else:
        raise ValueError(f"Unknown branch type: {branch_type}")

    print(f"  {branch_type} branch output: {ts_branch.output_dim}-dim")

    # --- Late Fusion ---
    model = LateFusionModel(
        ethgbert_branch=ethgbert_branch,
        gru_branch=ts_branch,
        extra_branches=None,
        num_classes=len(label2idx),
        fusion_type=cfg["fusion_type"],
        common_dim=cfg["common_dim"],
        dropout=cfg["dropout_rate"],
        freeze_ethgbert=cfg["freeze_ethgbert"],
        freeze_gru=cfg["freeze_gru"],
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    if cfg["freeze_ethgbert"]:
        print(f"  (ETH_GBert is FROZEN)")
    if cfg["freeze_gru"]:
        print(f"  (GRU is FROZEN)")

    model.to(DEVICE)
    return model


def load_or_init_model(cfg, label2idx, gcn_vocab_size, gcn_adj_list):
    """Load checkpoint nếu có, không thì tạo mới."""
    checkpoint_path = os.path.join(cfg["output_dir"], cfg["model_file"])

    if cfg.get("resume_from_checkpoint") and os.path.exists(checkpoint_path):
        print(f"[Model] Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model = build_model(cfg, label2idx, gcn_vocab_size, gcn_adj_list)

        # Load state dict (flexible: bỏ qua key không khớp)
        pretrained_dict = checkpoint["model_state"]
        model_dict = model.state_dict()
        matched = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        print(f"  Loaded {len(matched)}/{len(pretrained_dict)} parameter tensors")

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_valid_f1 = checkpoint.get("valid_f1", 0.0)
        return model, start_epoch, best_valid_f1
    else:
        model = build_model(cfg, label2idx, gcn_vocab_size, gcn_adj_list)
        return model, 0, 0.0


# ============================================================
# 6. EVALUATE
# ============================================================
def evaluate(model, gcn_adj_list, dataloader, cfg, loss_weight, num_classes,
             epoch, dataset_name="Valid"):
    """
    Evaluate model trên dataloader.
    Returns: (loss, accuracy, weighted_f1, precision, recall, attn_weights_avg)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_attn_weights = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(DEVICE) for t in batch)
            (input_ids, input_mask, segment_ids, y_prob,
             label_ids, gcn_swop_eye, timeseries) = batch

            ethgbert_inputs = {
                "vocab_adj_list": gcn_adj_list,
                "gcn_swop_eye": gcn_swop_eye,
                "input_ids": input_ids,
                "token_type_ids": segment_ids,
                "attention_mask": input_mask,
            }

            logits, attn_weights = model(ethgbert_inputs, timeseries)

            # Loss
            loss = F.cross_entropy(
                logits.view(-1, num_classes),
                label_ids,
                weight=loss_weight,
            )
            total_loss += loss.item()

            # Predictions
            _, predicted = torch.max(logits, -1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(label_ids.cpu().tolist())
            total_correct += predicted.eq(label_ids).sum().item()
            total_samples += len(label_ids)

            # Attention weights (nếu có)
            if attn_weights is not None:
                all_attn_weights.append(attn_weights.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    # Average attention weights
    attn_avg = None
    if all_attn_weights:
        attn_avg = np.concatenate(all_attn_weights, axis=0).mean(axis=0)

    print(f"\n--- {dataset_name} (Epoch {epoch}) ---")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print(f"  Loss: {total_loss:.4f} | Acc: {accuracy*100:.2f}% | "
          f"wF1: {weighted_f1*100:.2f}% | Pre: {precision*100:.2f}% | "
          f"Rec: {recall*100:.2f}%")

    if attn_avg is not None:
        branch_names = ["ETH_GBert", cfg["branch_type"]]
        attn_str = " | ".join(
            f"{name}: {w:.3f}" for name, w in zip(branch_names, attn_avg)
        )
        print(f"  Avg attention weights: [{attn_str}]")

    return total_loss, accuracy, weighted_f1, precision, recall, attn_avg


# ============================================================
# 7. MAIN TRAINING LOOP
# ============================================================
def train():
    args = parse_args()
    cfg = build_config(args)
    cfg["resume_from_checkpoint"] = args.load == 1
    print_config(cfg)

    # ----- Load ETH_GBert data -----
    ethgbert_objects, gcn_adj_list = load_ethgbert_data(cfg)
    example_info = build_examples(ethgbert_objects)

    label2idx = example_info["label2idx"]
    train_examples = example_info["train_examples"]
    valid_examples = example_info["valid_examples"]
    test_examples = example_info["test_examples"]
    num_classes = example_info["num_classes"]
    address_to_index = ethgbert_objects["address_to_index"]
    gcn_vocab_size = len(address_to_index)

    train_size = example_info["train_size"]
    valid_size = example_info["valid_size"]
    test_size = example_info["test_size"]

    # ----- Load GRU time-series data -----
    X_ts_train, X_ts_valid, X_ts_test = load_timeseries_data(
        cfg, train_size, valid_size, test_size
    )

    # Validate: check GRU input_size khớp
    actual_features = X_ts_train.shape[-1]
    if actual_features != cfg["gru_input_size"]:
        print(f"[WARN] --gru_input_size={cfg['gru_input_size']} nhưng data có "
              f"{actual_features} features. Tự động sửa.")
        cfg["gru_input_size"] = actual_features

    # ----- Validate program mode -----
    if args.validate_program:
        train_examples = [train_examples[0]]
        valid_examples = [valid_examples[0]]
        test_examples = [test_examples[0]]
        X_ts_train = X_ts_train[:1]
        X_ts_valid = X_ts_valid[:1]
        X_ts_test = X_ts_test[:1]

    # ----- Tokenizer -----
    print(f"\n[Tokenizer] Loading from: {cfg['bert_model_scale']}")
    tokenizer = BertTokenizer.from_pretrained(
        cfg["bert_model_scale"],
        do_lower_case=cfg["do_lower_case"],
    )

    # ----- DataLoaders -----
    print("\n[DataLoader] Building fusion dataloaders...")
    train_loader = build_fusion_dataloader(
        train_examples, X_ts_train, tokenizer, address_to_index,
        cfg["max_seq_length"], cfg["gcn_embedding_dim"],
        cfg["batch_size"], shuffle=True,
    )
    valid_loader = build_fusion_dataloader(
        valid_examples, X_ts_valid, tokenizer, address_to_index,
        cfg["max_seq_length"], cfg["gcn_embedding_dim"],
        cfg["batch_size"], shuffle=False,
    )
    test_loader = build_fusion_dataloader(
        test_examples, X_ts_test, tokenizer, address_to_index,
        cfg["max_seq_length"], cfg["gcn_embedding_dim"],
        cfg["batch_size"], shuffle=False,
    )

    # ----- Class weights -----
    train_classes_num, train_classes_weight = get_class_count_and_weight(
        example_info["train_y"], num_classes
    )
    loss_weight = torch.tensor(train_classes_weight, dtype=torch.float).to(DEVICE)

    # ----- Model -----
    gc.collect()
    model, start_epoch, best_valid_f1 = load_or_init_model(
        cfg, label2idx, gcn_vocab_size, gcn_adj_list
    )

    # ----- Optimizer -----
    total_train_steps = int(
        len(train_loader)
        / cfg["gradient_accumulation_steps"]
        * cfg["total_train_epochs"]
    )

    # Tách param groups: lr khác nhau cho các phần
    param_groups = []
    if not cfg["freeze_ethgbert"]:
        param_groups.append({
            "params": [p for p in model.ethgbert_branch.parameters() if p.requires_grad],
            "lr": cfg["learning_rate"],
        })
    if not cfg["freeze_gru"]:
        param_groups.append({
            "params": [p for p in model.gru_branch.parameters() if p.requires_grad],
            "lr": cfg["learning_rate"] * 10,  # GRU nhỏ hơn → lr lớn hơn
        })
    # Fusion + Classifier luôn train
    fusion_classifier_params = (
        list(model.fusion.parameters()) + list(model.classifier.parameters())
    )
    param_groups.append({
        "params": fusion_classifier_params,
        "lr": cfg["learning_rate"] * 5,
    })

    optimizer = BertAdam(
        param_groups,
        lr=cfg["learning_rate"],
        warmup=cfg["warmup_proportion"],
        t_total=total_train_steps,
        weight_decay=cfg["weight_decay"],
    )

    # ----- Print summary -----
    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  Train: {len(train_examples)} samples "
          f"({train_classes_num[0]} normal, {train_classes_num[1]} fraud)")
    print(f"  Valid: {len(valid_examples)} samples")
    print(f"  Test:  {len(test_examples)} samples")
    print(f"  Total train steps: {total_train_steps}")
    print(f"  Loss weights: {train_classes_weight}")
    print(f"{'='*60}\n")

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    train_start = time.time()
    global_step = 0

    best_valid_f1_epoch = -1
    best_test_f1 = 0.0
    best_test_f1_epoch = -1
    test_f1_when_valid_best = 0.0

    history = {
        "train_loss": [], "valid_loss": [], "test_loss": [],
        "valid_f1": [], "test_f1": [],
        "attn_weights": [],
    }

    for epoch in range(start_epoch, cfg["total_train_epochs"]):
        model.train()
        tr_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(DEVICE) for t in batch)
            (input_ids, input_mask, segment_ids, y_prob,
             label_ids, gcn_swop_eye, timeseries) = batch

            ethgbert_inputs = {
                "vocab_adj_list": gcn_adj_list,
                "gcn_swop_eye": gcn_swop_eye,
                "input_ids": input_ids,
                "token_type_ids": segment_ids,
                "attention_mask": input_mask,
            }

            logits, attn_weights = model(ethgbert_inputs, timeseries)

            # Loss
            loss = F.cross_entropy(
                logits.view(-1, num_classes),
                label_ids,
                weight=loss_weight,
            )

            if cfg["gradient_accumulation_steps"] > 1:
                loss = loss / cfg["gradient_accumulation_steps"]

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if step % 40 == 0:
                elapsed = (time.time() - train_start) / 60.0
                print(
                    f"Epoch {epoch} [{step}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.4f} | "
                    f"Time: {elapsed:.1f}m"
                )

        # ----- Evaluate -----
        print("\n" + "=" * 60)

        _, _, valid_f1, _, _, valid_attn = evaluate(
            model, gcn_adj_list, valid_loader, cfg, loss_weight,
            num_classes, epoch, "Valid"
        )

        _, _, test_f1, _, _, test_attn = evaluate(
            model, gcn_adj_list, test_loader, cfg, loss_weight,
            num_classes, epoch, "Test"
        )

        # Track history
        history["train_loss"].append(tr_loss)
        history["valid_f1"].append(valid_f1)
        history["test_f1"].append(test_f1)
        if valid_attn is not None:
            history["attn_weights"].append(valid_attn.tolist())

        elapsed = (time.time() - train_start) / 60.0
        print(f"\nEpoch {epoch} done | Train Loss: {tr_loss:.4f} | "
              f"Valid wF1: {valid_f1*100:.2f}% | Test wF1: {test_f1*100:.2f}% | "
              f"Time: {elapsed:.1f}m")

        # Track best test
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_test_f1_epoch = epoch

        # Save best model (by valid F1)
        if valid_f1 > best_valid_f1:
            save_path = os.path.join(cfg["output_dir"], cfg["model_file"])
            to_save = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "valid_f1": valid_f1,
                "test_f1": test_f1,
                "config": cfg,
                "attn_weights": valid_attn.tolist() if valid_attn is not None else None,
            }
            torch.save(to_save, save_path)
            best_valid_f1 = valid_f1
            best_valid_f1_epoch = epoch
            test_f1_when_valid_best = test_f1
            print(f"  >>> Saved best model (Valid wF1={valid_f1*100:.2f}%) → {save_path}")

        print("=" * 60)

    # ============================================================
    # FINAL REPORT
    # ============================================================
    total_time = (time.time() - train_start) / 60.0
    print("\n" + "#" * 60)
    print("  TRAINING COMPLETE")
    print("#" * 60)
    print(f"  Total time:                  {total_time:.1f} minutes")
    print(f"  Best Valid wF1:              {best_valid_f1*100:.2f}% (epoch {best_valid_f1_epoch})")
    print(f"  Test wF1 when valid best:    {test_f1_when_valid_best*100:.2f}%")
    print(f"  Best Test wF1 (absolute):    {best_test_f1*100:.2f}% (epoch {best_test_f1_epoch})")

    if history["attn_weights"]:
        final_attn = history["attn_weights"][-1]
        print(f"\n  Final attention weights:")
        print(f"    ETH_GBert:        {final_attn[0]:.3f}")
        print(f"    {cfg['branch_type']:17s} {final_attn[1]:.3f}")

    print("#" * 60)

    # Save history
    history_path = os.path.join(cfg["output_dir"], "fusion_training_history.pkl")
    with open(history_path, "wb") as f:
        pkl.dump(history, f)
    print(f"\nTraining history saved to: {history_path}")


if __name__ == "__main__":
    train()