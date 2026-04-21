"""
LateFusion.py

Kiến trúc Late Fusion kết hợp ETH_GBert (text+graph) với GRU (time-series),
và mở rộng được cho N models bất kỳ.

Pipeline:
    ETH_GBert  ──→  pooled_output (768-dim)  ──┐
                                                ├──→  Attention Fusion  ──→  Classifier  ──→  0/1
    GRU        ──→  hidden_state  (64-dim)   ──┘

Cách dùng:
    model = LateFusionModel(ethgbert, gru_branch, num_classes=2)
    logits = model(ethgbert_inputs, gru_inputs)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. GRU BRANCH — trích hidden state (không có classifier)
# ============================================================
class GRUBranch(nn.Module):
    """
    GRU cho time-series features.
    Output: hidden representation (KHÔNG qua classifier).
    """

    def __init__(self, input_size=10, hidden_size=64,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        self.output_dim = hidden_size

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        return: (batch, hidden_size)
        """
        out, _ = self.gru(x)
        return out[:, -1, :]  # last hidden state


# ============================================================
# 1b. GRU + ATTENTION BRANCH
# ============================================================
class GRUAttentionBranch(nn.Module):
    """
    GRU + Self-Attention: thay vì lấy last hidden state,
    dùng attention tổng hợp TẤT CẢ hidden states.
    """

    def __init__(self, input_size=10, hidden_size=64,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        # Attention: score mỗi timestep
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.output_dim = hidden_size

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        return: (batch, hidden_size)
        """
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size)

        # Attention weights
        attn_scores = self.attention(gru_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum
        context = (attn_weights * gru_out).sum(dim=1)  # (batch, hidden_size)
        return context


# ============================================================
# 1c. BiGRU BRANCH
# ============================================================
class BiGRUBranch(nn.Module):
    """
    Bidirectional GRU: đọc sequence cả 2 chiều.
    Output dim = hidden_size * 2 (concat forward + backward).
    """

    def __init__(self, input_size=10, hidden_size=64,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.output_dim = hidden_size * 2

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        return: (batch, hidden_size * 2)
        """
        out, _ = self.gru(x)  # (batch, seq_len, hidden_size*2)
        return out[:, -1, :]  # last hidden state (forward + backward)


# ============================================================
# 1d. TCN BRANCH (Temporal Convolutional Network)
# ============================================================
class TCNBlock(nn.Module):
    """1 block của TCN: dilated causal conv + residual."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(
            self.conv1, self.relu, self.dropout,
            self.conv2, self.relu, self.dropout,
        )
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        """x: (batch, channels, seq_len)"""
        out = self.net(x)
        # Causal: trim future (padding thêm ở bên phải)
        out = out[:, :, :x.size(2)]
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)


class TCNBranch(nn.Module):
    """
    Temporal Convolutional Network cho time-series.
    Dùng dilated causal convolutions để capture long-range dependencies.
    """

    def __init__(self, input_size=10, hidden_size=64,
                 num_layers=4, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            layers.append(TCNBlock(in_ch, hidden_size, kernel_size,
                                   dilation=2**i, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_size

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        return: (batch, hidden_size)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        out = self.network(x)   # (batch, hidden_size, seq_len)
        return out[:, :, -1]    # last timestep: (batch, hidden_size)


# ============================================================
# 1e. CNN + GRU BRANCH
# ============================================================
class CNNGRUBranch(nn.Module):
    """
    CNN extract local patterns → GRU capture sequential dependencies.
    CNN: 1D conv trên time axis.
    """

    def __init__(self, input_size=10, hidden_size=64,
                 cnn_channels=32, kernel_size=3,
                 num_layers=2, dropout=0.3):
        super().__init__()
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # GRU on top of CNN features
        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_dim = hidden_size

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        return: (batch, hidden_size)
        """
        # CNN: (batch, input_size, seq_len)
        cnn_in = x.transpose(1, 2)
        cnn_out = self.cnn(cnn_in)  # (batch, cnn_channels, seq_len)

        # GRU: (batch, seq_len, cnn_channels)
        gru_in = cnn_out.transpose(1, 2)
        gru_out, _ = self.gru(gru_in)  # (batch, seq_len, hidden_size)
        return gru_out[:, -1, :]  # last hidden state


# ============================================================
# 2. ETH_GBert WRAPPER — trích pooled_output (không qua classifier)
# ============================================================
class ETHGBertBranch(nn.Module):
    """
    Wrapper cho ETH_GBertModel.
    Trích pooled_output (768-dim) TRƯỚC classifier.
    """

    def __init__(self, ethgbert_model):
        super().__init__()
        self.ethgbert = ethgbert_model
        self.output_dim = self.ethgbert.config.hidden_size  # 768

    def forward(self, vocab_adj_list, gcn_swop_eye,
                input_ids, token_type_ids=None, attention_mask=None):
        """
        Chạy ETH_GBert nhưng dừng lại ở pooled_output,
        không đi qua self.ethgbert.classifier.
        """
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Embedding (BERT + GCN fusion bên trong)
        embedding_output = self.ethgbert.embeddings(
            vocab_adj_list,
            gcn_swop_eye,
            input_ids,
            token_type_ids,
            attention_mask,
        )

        # Encoder
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.ethgbert.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.ethgbert.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=False,
            head_mask=[None] * self.ethgbert.config.num_hidden_layers,
        )

        sequence_output = encoded_layers[-1]

        # Pooler: lấy [CLS] token → Linear → Tanh → (batch, 768)
        pooled_output = self.ethgbert.pooler(sequence_output)

        return pooled_output  # (batch, 768)


# ============================================================
# 3. ATTENTION FUSION — fuse N nhánh bằng attention
# ============================================================
class AttentionFusion(nn.Module):
    """
    Attention-based fusion cho N branches có kích thước khác nhau.

    Bước 1: Project mỗi branch về cùng common_dim
    Bước 2: Attention weights cho mỗi branch
    Bước 3: Weighted sum → fused representation

    Ưu điểm so với concat:
    - Tự học branch nào quan trọng hơn
    - Không phụ thuộc vào số lượng branch (mở rộng dễ)
    - Không bị bias bởi branch có dim lớn hơn
    """

    def __init__(self, branch_dims, common_dim=256, dropout=0.2):
        """
        Args:
            branch_dims: list[int], e.g. [768, 64] cho [ETH_GBert, GRU]
            common_dim:  chiều không gian chung để fuse
            dropout:     dropout rate
        """
        super().__init__()
        self.n_branches = len(branch_dims)
        self.common_dim = common_dim

        # Project mỗi branch về common_dim
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, common_dim),
                nn.LayerNorm(common_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for dim in branch_dims
        ])

        # Attention: query = concat tất cả projected → score cho mỗi branch
        self.attention = nn.Sequential(
            nn.Linear(common_dim * self.n_branches, common_dim),
            nn.Tanh(),
            nn.Linear(common_dim, self.n_branches),
        )

        self.output_dim = common_dim

    def forward(self, branch_outputs):
        """
        Args:
            branch_outputs: list[Tensor], mỗi tensor shape (batch, branch_dim_i)
        Returns:
            fused: (batch, common_dim)
            attention_weights: (batch, n_branches) — để debug/visualize
        """
        # Project tất cả về common_dim
        projected = [
            proj(feat) for proj, feat in zip(self.projections, branch_outputs)
        ]
        # projected[i] shape: (batch, common_dim)

        # Stack → (batch, n_branches, common_dim)
        stacked = torch.stack(projected, dim=1)

        # Attention weights
        concat_all = torch.cat(projected, dim=-1)  # (batch, common_dim * n_branches)
        attn_scores = self.attention(concat_all)    # (batch, n_branches)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, n_branches)

        # Weighted sum
        # attn_weights: (batch, n_branches, 1) × stacked: (batch, n_branches, common_dim)
        fused = (attn_weights.unsqueeze(-1) * stacked).sum(dim=1)  # (batch, common_dim)

        return fused, attn_weights


# ============================================================
# 4. CONCAT FUSION — fuse đơn giản bằng concat + MLP
# ============================================================
class ConcatFusion(nn.Module):
    """
    Phương pháp đơn giản: concat tất cả branches → MLP.
    Dùng khi không cần attention interpretability.
    """

    def __init__(self, branch_dims, common_dim=256, dropout=0.2):
        super().__init__()
        total_dim = sum(branch_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_dim = common_dim

    def forward(self, branch_outputs):
        concatenated = torch.cat(branch_outputs, dim=-1)
        fused = self.fusion(concatenated)
        return fused, None  # None cho attention_weights (không có)


# ============================================================
# 5. LATE FUSION MODEL — kết hợp tất cả
# ============================================================
class LateFusionModel(nn.Module):
    """
    Late Fusion: ETH_GBert + GRU (+ thêm bao nhiêu branch cũng được).

    Kiến trúc:
        Branch 1 (ETH_GBert): text+graph → pooled_output (768)  ─┐
        Branch 2 (GRU):       timeseries → hidden_state (64)    ─┤
        Branch N (...):       ...        → ... (dim_n)           ─┤
                                                                  ↓
                                                         Attention Fusion
                                                                  ↓
                                                            Classifier
                                                                  ↓
                                                               0 / 1
    """

    def __init__(
        self,
        ethgbert_branch,
        gru_branch,
        extra_branches=None,
        num_classes=2,
        fusion_type="attention",
        common_dim=256,
        dropout=0.2,
        freeze_ethgbert=False,
        freeze_gru=False,
    ):
        """
        Args:
            ethgbert_branch: ETHGBertBranch instance
            gru_branch:      GRUBranch instance
            extra_branches:  list[nn.Module] — thêm branch nào cũng được,
                             mỗi branch cần có .output_dim
            num_classes:     số class (2: normal/phishing)
            fusion_type:     "attention" hoặc "concat"
            common_dim:      chiều fused representation
            dropout:         dropout rate
            freeze_ethgbert: đóng băng ETH_GBert (chỉ train fusion + classifier)
            freeze_gru:      đóng băng GRU
        """
        super().__init__()

        # Branches
        self.ethgbert_branch = ethgbert_branch
        self.gru_branch = gru_branch
        self.extra_branches = nn.ModuleList(extra_branches or [])

        # Freeze nếu cần
        if freeze_ethgbert:
            for p in self.ethgbert_branch.parameters():
                p.requires_grad = False
        if freeze_gru:
            for p in self.gru_branch.parameters():
                p.requires_grad = False

        # Tính dims cho tất cả branches
        branch_dims = [
            ethgbert_branch.output_dim,   # 768
            gru_branch.output_dim,        # 64
        ]
        for branch in self.extra_branches:
            branch_dims.append(branch.output_dim)

        # Fusion layer
        if fusion_type == "attention":
            self.fusion = AttentionFusion(branch_dims, common_dim, dropout)
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(branch_dims, common_dim, dropout)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.output_dim, self.fusion.output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.fusion.output_dim // 2, num_classes),
        )

    def forward(self, ethgbert_inputs, gru_input, extra_inputs=None):
        """
        Args:
            ethgbert_inputs: dict với keys:
                - vocab_adj_list
                - gcn_swop_eye
                - input_ids
                - token_type_ids (optional)
                - attention_mask (optional)
            gru_input: Tensor (batch, seq_len, input_size)
            extra_inputs: list[Tensor] cho extra_branches (nếu có)

        Returns:
            logits: (batch, num_classes)
            attn_weights: (batch, n_branches) hoặc None
        """
        # Branch 1: ETH_GBert → (batch, 768)
        ethgbert_out = self.ethgbert_branch(**ethgbert_inputs)

        # Branch 2: GRU → (batch, 64)
        gru_out = self.gru_branch(gru_input)

        # Gom tất cả branch outputs
        branch_outputs = [ethgbert_out, gru_out]

        # Extra branches (nếu có)
        if extra_inputs is not None:
            for branch, inp in zip(self.extra_branches, extra_inputs):
                branch_outputs.append(branch(inp))

        # Fusion
        fused, attn_weights = self.fusion(branch_outputs)

        # Classify
        logits = self.classifier(fused)

        return logits, attn_weights


# ============================================================
# 6. LSTM BRANCH (ví dụ thêm branch thứ 3)
# ============================================================
class LSTMBranch(nn.Module):
    """
    LSTM cho time-series features.
    Output: hidden representation (KHÔNG qua classifier).
    """

    def __init__(self, input_size=10, hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        self.output_dim = hidden_size

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        return: (batch, hidden_size)
        """
        out, (h_n, c_n) = self.lstm(x)
        return out[:, -1, :]  # last hidden state


# ============================================================
# 7. EXAMPLE USAGE — chạy được thật với dữ liệu thật
# ============================================================
def example_usage():
    """
    Ví dụ đầy đủ: load data → build model → forward pass → in kết quả.

    Yêu cầu:
        - data/preprocessed/Dataset/ chứa các file preprocessed
        - data/preprocessed/Dataset/X_timeseries.npy (hoặc 3 file riêng)
        - ETH_GBert.py, utils.py, env_config.py ở cùng thư mục
    """
    import os
    import pickle as pkl
    import numpy as np
    import scipy.sparse as sp

    from ETH_GBert import ETH_GBertModel
    from pytorch_pretrained_bert.tokenization import BertTokenizer
    from env_config import env_config
    from utils import (
        InputExample,
        example2feature,
        sparse_scipy2torch,
    )

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {DEVICE}")

    # =============================================
    # Config
    # =============================================
    DATASET = "Dataset"
    DATA_DIR = f"data/preprocessed/{DATASET}"
    GCN_EMBEDDING_DIM = 16
    MAX_SEQ_LENGTH = 200 + GCN_EMBEDDING_DIM
    NUM_CLASSES = 2
    BATCH_SIZE = 4  # nhỏ cho demo

    # BERT path: dùng domain-adapted nếu có, không thì dùng bert-base-uncased
    BERT_MODEL_SCALE = "bert-base-uncased"
    if env_config.TRANSFORMERS_OFFLINE == 1:
        BERT_MODEL_SCALE = os.path.join(
            env_config.HUGGING_LOCAL_MODEL_FILES_PATH,
            f"hf-maintainers_{BERT_MODEL_SCALE}",
        )

    # =============================================
    # Bước 1: Load ETH_GBert preprocessed data
    # =============================================
    print("\n[1/6] Loading ETH_GBert preprocessed data...")

    def load_pkl(name):
        path = os.path.join(DATA_DIR, f"data_{DATASET}.{name}")
        with open(path, "rb") as f:
            return pkl.load(f, encoding="latin1")

    labels_list = load_pkl("labels")           # [label2idx, idx2label]
    train_y = load_pkl("train_y")
    valid_y = load_pkl("valid_y")
    test_y = load_pkl("test_y")
    train_y_prob = load_pkl("train_y_prob")
    valid_y_prob = load_pkl("valid_y_prob")
    test_y_prob = load_pkl("test_y_prob")
    shuffled_clean_docs = load_pkl("shuffled_clean_docs")
    address_to_index = load_pkl("address_to_index")

    label2idx = labels_list[0]
    idx2label = labels_list[1]
    gcn_vocab_size = len(address_to_index)

    train_size = len(train_y)
    valid_size = len(valid_y)
    test_size = len(test_y)

    print(f"  label2idx: {label2idx}")
    print(f"  Train: {train_size}, Valid: {valid_size}, Test: {test_size}")
    print(f"  GCN vocab size: {gcn_vocab_size}")

    # =============================================
    # Bước 2: Load GCN adjacency matrix
    # =============================================
    print("\n[2/6] Loading GCN adjacency matrix...")
    npz = np.load(os.path.join(DATA_DIR, "norm_adj_coo.npz"))
    a_hat = sp.coo_matrix(
        (npz["data"], (npz["row"], npz["col"])),
        shape=tuple(npz["shape"]),
    )
    gcn_adj_list = [sparse_scipy2torch(a_hat).to(DEVICE)]
    print(f"  Adjacency: shape={a_hat.shape}, nnz={a_hat.nnz}")

    # =============================================
    # Bước 3: Load GRU time-series data
    # =============================================
    print("\n[3/6] Loading GRU time-series data...")
    ts_train_path = os.path.join(DATA_DIR, "X_timeseries_train.npy")
    ts_all_path = os.path.join(DATA_DIR, "X_timeseries.npy")

    if os.path.exists(ts_train_path):
        X_ts_train = np.load(ts_train_path)
        X_ts_valid = np.load(os.path.join(DATA_DIR, "X_timeseries_valid.npy"))
        X_ts_test = np.load(os.path.join(DATA_DIR, "X_timeseries_test.npy"))
    elif os.path.exists(ts_all_path):
        X_ts_all = np.load(ts_all_path)
        X_ts_train = X_ts_all[:train_size]
        X_ts_valid = X_ts_all[train_size: train_size + valid_size]
        X_ts_test = X_ts_all[train_size + valid_size:]
    else:
        raise FileNotFoundError(
            f"Không tìm thấy time-series data trong {DATA_DIR}.\n"
            f"Cần: X_timeseries.npy hoặc X_timeseries_train/valid/test.npy"
        )

    gru_input_size = X_ts_train.shape[-1]
    print(f"  Train: {X_ts_train.shape}, Valid: {X_ts_valid.shape}, Test: {X_ts_test.shape}")
    print(f"  GRU input_size (features per timestep): {gru_input_size}")

    # =============================================
    # Bước 4: Build model (2 branches: ETH_GBert + GRU)
    # =============================================
    print("\n[4/6] Building LateFusionModel...")

    # Branch 1: ETH_GBert
    ethgbert_model = ETH_GBertModel.from_pretrained(
        BERT_MODEL_SCALE,
        gcn_adj_dim=gcn_vocab_size,
        gcn_adj_num=len(gcn_adj_list),
        gcn_embedding_dim=GCN_EMBEDDING_DIM,
        num_labels=NUM_CLASSES,
    )
    ethgbert_branch = ETHGBertBranch(ethgbert_model)
    print(f"  ETH_GBert branch: output_dim={ethgbert_branch.output_dim}")

    # Branch 2: GRU
    gru_branch = GRUBranch(
        input_size=gru_input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
    )
    print(f"  GRU branch: output_dim={gru_branch.output_dim}")

    # Late Fusion (attention)
    model = LateFusionModel(
        ethgbert_branch=ethgbert_branch,
        gru_branch=gru_branch,
        extra_branches=None,
        num_classes=NUM_CLASSES,
        fusion_type="attention",
        common_dim=256,
        dropout=0.2,
        freeze_ethgbert=False,
        freeze_gru=False,
    )
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # =============================================
    # Bước 5: Tạo 1 mini-batch và chạy forward pass
    # =============================================
    print(f"\n[5/6] Forward pass demo (batch_size={BATCH_SIZE})...")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        BERT_MODEL_SCALE,
        do_lower_case=True,
    )

    # Tạo examples cho batch đầu tiên
    y = np.hstack((train_y, valid_y, test_y))
    y_prob = np.vstack((train_y_prob, valid_y_prob, test_y_prob))

    batch_examples = []
    for i in range(BATCH_SIZE):
        batch_examples.append(
            InputExample(i, shuffled_clean_docs[i].strip(),
                         confidence=y_prob[i], label=y[i])
        )

    # Tokenize + tạo features
    batch_features = []
    for ex in batch_examples:
        feat = example2feature(ex, tokenizer, address_to_index,
                               MAX_SEQ_LENGTH, GCN_EMBEDDING_DIM)
        batch_features.append(feat)

    # Pad thủ công (giống CorpusDataset.pad)
    seq_lens = [len(f.input_ids) for f in batch_features]
    max_len = max(seq_lens)

    input_ids_padded = [
        f.input_ids + [0] * (max_len - len(f.input_ids))
        for f in batch_features
    ]
    input_mask_padded = [
        f.input_mask + [0] * (max_len - len(f.input_mask))
        for f in batch_features
    ]
    segment_ids_padded = [
        f.segment_ids + [0] * (max_len - len(f.segment_ids))
        for f in batch_features
    ]
    gcn_ids_padded = [
        [-1] + f.gcn_vocab_ids + [-1] * (max_len - len(f.gcn_vocab_ids) - 1)
        for f in batch_features
    ]

    input_ids = torch.tensor(input_ids_padded, dtype=torch.long).to(DEVICE)
    input_mask = torch.tensor(input_mask_padded, dtype=torch.long).to(DEVICE)
    segment_ids = torch.tensor(segment_ids_padded, dtype=torch.long).to(DEVICE)
    label_ids = torch.tensor([f.label_id for f in batch_features], dtype=torch.long).to(DEVICE)

    # GCN swap eye
    gcn_ids_flat = np.array(gcn_ids_padded).reshape(-1)
    gcn_swop_eye = torch.eye(gcn_vocab_size + 1)[gcn_ids_flat][:, :-1]
    gcn_swop_eye = gcn_swop_eye.view(BATCH_SIZE, -1, gcn_vocab_size).transpose(1, 2).to(DEVICE)

    # GRU input
    gru_input = torch.tensor(X_ts_train[:BATCH_SIZE], dtype=torch.float32).to(DEVICE)

    # Forward pass
    model.eval()
    with torch.no_grad():
        ethgbert_inputs = {
            "vocab_adj_list": gcn_adj_list,
            "gcn_swop_eye": gcn_swop_eye,
            "input_ids": input_ids,
            "token_type_ids": segment_ids,
            "attention_mask": input_mask,
        }

        logits, attn_weights = model(ethgbert_inputs, gru_input)

    print(f"  logits shape:       {logits.shape}")       # (batch, 2)
    print(f"  logits:\n{logits}")
    print(f"  predictions:        {logits.argmax(dim=-1).tolist()}")
    print(f"  true labels:        {label_ids.tolist()}")

    if attn_weights is not None:
        print(f"  attn_weights shape: {attn_weights.shape}")  # (batch, 2)
        print(f"  attn_weights (mỗi sample → [ETH_GBert, GRU]):")
        for i in range(BATCH_SIZE):
            print(f"    Sample {i}: ETH_GBert={attn_weights[i][0]:.3f}, "
                  f"GRU={attn_weights[i][1]:.3f}")

    # =============================================
    # Bước 6 (bonus): Ví dụ thêm LSTM branch thứ 3
    # =============================================
    print(f"\n[6/6] Demo 3-branch model (ETH_GBert + GRU + LSTM)...")

    lstm_branch = LSTMBranch(
        input_size=gru_input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
    )
    print(f"  LSTM branch: output_dim={lstm_branch.output_dim}")

    model_3branch = LateFusionModel(
        ethgbert_branch=ethgbert_branch,
        gru_branch=gru_branch,
        extra_branches=[lstm_branch],
        num_classes=NUM_CLASSES,
        fusion_type="attention",
        common_dim=256,
    )
    model_3branch.to(DEVICE)
    model_3branch.eval()

    total_3 = sum(p.numel() for p in model_3branch.parameters())
    print(f"  3-branch total params: {total_3:,}")

    # Forward pass 3-branch
    lstm_input = gru_input.clone()  # cùng time-series data cho demo

    with torch.no_grad():
        logits_3, attn_3 = model_3branch(
            ethgbert_inputs, gru_input, extra_inputs=[lstm_input]
        )

    print(f"  logits shape:       {logits_3.shape}")
    print(f"  predictions:        {logits_3.argmax(dim=-1).tolist()}")

    if attn_3 is not None:
        print(f"  attn_weights (mỗi sample → [ETH_GBert, GRU, LSTM]):")
        for i in range(BATCH_SIZE):
            print(f"    Sample {i}: ETH_GBert={attn_3[i][0]:.3f}, "
                  f"GRU={attn_3[i][1]:.3f}, LSTM={attn_3[i][2]:.3f}")

    print("\n[DONE] LateFusion chạy thành công!")


if __name__ == "__main__":
    example_usage()