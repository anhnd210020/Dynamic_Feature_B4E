"""
preprocess_b4e.py

Bước 2: Preprocessing all-in-one cho B4E dataset.
Tạo tất cả files cần thiết cho ETH-GBert + MLM + timeseries.

Input:
    Dataset/MulDiGraph.pkl  (từ build_b4e_graph.py)

Output (vào data/preprocessed/B4E/):
    - train.tsv, dev.tsv, test.tsv          (text cho BERT)
    - data_B4E.labels, .train_y, .valid_y, .test_y, ...
    - data_B4E.shuffled_clean_docs
    - data_B4E.address_to_index
    - norm_adj_coo.npz                      (GCN adjacency)
    - X_timeseries_train/valid/test.npy     (GRU timeseries)
    - address_alignment.pkl                 (verify alignment)

Pipeline:
    1) Load graph → extract transactions per account
    2) Add time n-grams (2-5)
    3) Balance 2:1 (normal:phishing)
    4) Generate text sentences
    5) Split 80/10/10 (stratified)
    6) Save TSVs
    7) Replay BERT_text_data.py shuffle → xác định thứ tự cuối cùng
    8) Build weighted adjacency matrix + normalize
    9) Build timeseries (aligned)
    10) Save all preprocessed files

Cách chạy:
    cd ~/Dynamic_Feature_B4E/Dataset
    python preprocess_b4e.py
    python preprocess_b4e.py --ratio 1   # 1:1 thay vì 2:1
"""

import argparse
import os
import pickle
import random
import time
from datetime import datetime

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ============================================================
# 0. CONFIG
# ============================================================
def parse_args():
    # Paths relative to Dataset/ folder
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Dataset/
    project_dir = os.path.dirname(script_dir)                # Dynamic_Feature_B4E/

    p = argparse.ArgumentParser(description="Preprocess B4E for ETH-GBert")
    p.add_argument("--graph_file", type=str,
                    default=os.path.join(script_dir, "MulDiGraph.pkl"))
    p.add_argument("--output_dir", type=str,
                    default=os.path.join(project_dir, "data", "preprocessed", "B4E"))
    p.add_argument("--dataset_name", type=str, default="B4E",
                    help="Tên dataset (dùng cho file prefix)")
    p.add_argument("--ratio", type=int, default=2,
                    help="Normal:Phishing ratio")
    p.add_argument("--max_seq_len", type=int, default=50,
                    help="Max timesteps cho timeseries")
    p.add_argument("--seed", type=int, default=44)
    return p.parse_args()


# ============================================================
# 1. EXTRACT TRANSACTIONS
# ============================================================
def extract_transactions(G):
    """Gom IN+OUT edges, sort, dedup per account. Chỉ giữ labeled accounts."""
    print("[1/10] Extracting transactions per labeled account...")

    accounts = {}
    skipped_unlabeled = 0

    for u, v, data in tqdm(G.edges(data=True), desc="Edges",
                           total=G.number_of_edges()):
        amount = float(data.get("amount", 0.0))
        ts = int(data.get("timestamp", 0))
        tag_u = int(G.nodes[u].get("isp", -1))
        tag_v = int(G.nodes[v].get("isp", -1))

        # OUT edge: from u
        if tag_u in (0, 1):
            accounts.setdefault(u, []).append({
                "tag": tag_u, "from_address": u, "to_address": v,
                "amount": amount, "timestamp": ts, "in_out": 1,
            })

        # IN edge: to v
        if tag_v in (0, 1):
            accounts.setdefault(v, []).append({
                "tag": tag_v, "from_address": u, "to_address": v,
                "amount": amount, "timestamp": ts, "in_out": 0,
            })

    # Sort + dedup
    for addr, txs in accounts.items():
        txs.sort(key=lambda x: int(x.get("timestamp", 0)))
        seen = set()
        deduped = []
        for t in txs:
            key = (t["from_address"], t["to_address"],
                   float(t["amount"]), int(t["timestamp"]), int(t["in_out"]))
            if key not in seen:
                seen.add(key)
                deduped.append(t)
        accounts[addr] = deduped

    n0 = sum(1 for a, txs in accounts.items() if txs and txs[0].get("tag") == 0)
    n1 = sum(1 for a, txs in accounts.items() if txs and txs[0].get("tag") == 1)
    print(f"  Accounts: normal={n0:,}, phishing={n1:,}, total={len(accounts):,}")

    return accounts


# ============================================================
# 2. ADD TIME N-GRAMS
# ============================================================
def add_time_ngrams(accounts, max_n=5):
    print("[2/10] Adding time n-grams...")
    for addr, txs in accounts.items():
        for n in range(2, max_n + 1):
            key = f"{n}-gram"
            for i in range(len(txs)):
                if i < n - 1:
                    txs[i][key] = 0.0
                else:
                    txs[i][key] = float(
                        txs[i]["timestamp"] - txs[i - (n - 1)]["timestamp"]
                    )
    return accounts


# ============================================================
# 3. BALANCE
# ============================================================
def balance_accounts(accounts, ratio, seed):
    print(f"[3/10] Balancing (ratio={ratio}:1)...")
    rng = random.Random(seed)

    tag1, tag0 = [], []
    for addr, txs in accounts.items():
        if not txs:
            continue
        tag = int(txs[0].get("tag", -1))
        if tag == 1:
            tag1.append(addr)
        elif tag == 0:
            tag0.append(addr)

    print(f"  Before: normal={len(tag0):,}, phishing={len(tag1):,}")

    sample_n = min(len(tag0), ratio * len(tag1))
    selected_normal = rng.sample(tag0, sample_n)
    selected = tag1 + selected_normal
    rng.shuffle(selected)

    print(f"  After:  normal={len(selected_normal):,}, phishing={len(tag1):,}, "
          f"total={len(selected):,}")
    return selected


# ============================================================
# 4. GENERATE TEXT
# ============================================================
DESC_FIELDS = ["in_out", "amount", "2-gram", "3-gram", "4-gram", "5-gram"]


def account_to_sentence(txs):
    if not txs:
        return 0, "."
    tag = int(txs[0].get("tag", 0))
    parts = []
    for t in txs:
        desc = " ".join(f"{k}:{t[k]}" for k in DESC_FIELDS if k in t)
        if desc:
            parts.append(desc)
    return tag, "  ".join(parts) + "."


def sanitize(s):
    return s.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


# ============================================================
# 5. SPLIT 80/10/10
# ============================================================
def split_dataset(addresses, tags, seed):
    print("[5/10] Splitting 80/10/10 (stratified)...")
    y = np.array(tags)
    indices = np.arange(len(y))

    idx_train, idx_tmp = train_test_split(
        indices, train_size=0.8, random_state=seed, stratify=y,
    )
    idx_valid, idx_test = train_test_split(
        idx_tmp, test_size=0.5, random_state=seed, stratify=y[idx_tmp],
    )

    print(f"  Train: {len(idx_train)} (fraud={y[idx_train].sum()})")
    print(f"  Valid: {len(idx_valid)} (fraud={y[idx_valid].sum()})")
    print(f"  Test:  {len(idx_test)} (fraud={y[idx_test].sum()})")

    return idx_train, idx_valid, idx_test


# ============================================================
# 6. SAVE TSVs
# ============================================================
def save_tsvs(tags, sentences, idx_train, idx_valid, idx_test, output_dir):
    print("[6/10] Saving TSVs...")

    with open(os.path.join(output_dir, "train.tsv"), "w", encoding="utf-8") as f:
        f.write("label\tsentence\n")
        for i in idx_train:
            f.write(f"{tags[i]}\t{sanitize(sentences[i])}\n")

    with open(os.path.join(output_dir, "dev.tsv"), "w", encoding="utf-8") as f:
        f.write("label\tsentence\n")
        for i in idx_valid:
            f.write(f"{tags[i]}\t{sanitize(sentences[i])}\n")

    with open(os.path.join(output_dir, "test.tsv"), "w", encoding="utf-8") as f:
        f.write("index\tsentence\n")
        for j, i in enumerate(idx_test):
            f.write(f"{j}\t{sanitize(sentences[i])}\n")

    print(f"  train.tsv: {len(idx_train)}, dev.tsv: {len(idx_valid)}, "
          f"test.tsv: {len(idx_test)}")


# ============================================================
# 7. REPLAY BERT_text_data.py SHUFFLE
# ============================================================
def replay_bert_shuffle(addresses, tags, idx_train_tsv, idx_valid_tsv, seed):
    """
    Replay EXACT shuffle của BERT_text_data.py:
        np.random.seed(44); random.seed(44)
        train_valid_df = shuffle(train_valid_df)   # train.tsv
        test_df = shuffle(test_df)                 # dev.tsv
        valid_size = int(len(train_valid) * 0.05)
        train_size = len(train_valid) - valid_size
    """
    print("[7/10] Replaying BERT_text_data.py shuffle...")

    tv_addrs = [addresses[i] for i in idx_train_tsv]
    t_addrs = [addresses[i] for i in idx_valid_tsv]
    tv_tags = [tags[i] for i in idx_train_tsv]
    t_tags = [tags[i] for i in idx_valid_tsv]

    n_tv, n_t = len(tv_addrs), len(t_addrs)

    np.random.seed(seed)
    random.seed(seed)

    perm_tv = np.random.permutation(n_tv)
    shuffled_tv_addrs = [tv_addrs[i] for i in perm_tv]
    shuffled_tv_tags = [tv_tags[i] for i in perm_tv]

    perm_t = np.random.permutation(n_t)
    shuffled_t_addrs = [t_addrs[i] for i in perm_t]
    shuffled_t_tags = [t_tags[i] for i in perm_t]

    final_addrs = shuffled_tv_addrs + shuffled_t_addrs
    final_tags = shuffled_tv_tags + shuffled_t_tags

    valid_size = int(n_tv * 0.05)
    train_size = n_tv - valid_size
    test_size = n_t

    print(f"  Final order: train={train_size}, valid={valid_size}, test={test_size}")

    return {
        "final_addresses": final_addrs,
        "final_tags": final_tags,
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size,
    }


# ============================================================
# 8. BUILD ADJACENCY MATRIX
# ============================================================
def calculate_weight(tx):
    w = 0.0
    w += float(tx.get("2-gram", 0)) * 0.1
    w += float(tx.get("3-gram", 0)) * 0.2
    w += float(tx.get("4-gram", 0)) * 0.3
    w += float(tx.get("5-gram", 0)) * 0.4
    return w


def build_adjacency(accounts, selected_addresses, output_dir, seed):
    """Build normalized adjacency matrix (giống adjust_matrix.py)."""
    print("[8/10] Building adjacency matrix...")

    random.seed(seed)
    np.random.seed(seed)

    # Address → index mapping (sorted for stability)
    addresses_sorted = sorted(selected_addresses)
    address_to_index = {addr: idx for idx, addr in enumerate(addresses_sorted)}
    n = len(addresses_sorted)

    # Build weighted edges
    rows, cols, vals = [], [], []
    for addr in selected_addresses:
        if addr not in accounts:
            continue
        for t in accounts[addr]:
            fa = t.get("from_address")
            ta = t.get("to_address")
            if fa in address_to_index and ta in address_to_index:
                w = calculate_weight(t)
                if w != 0.0:
                    rows.append(address_to_index[fa])
                    cols.append(address_to_index[ta])
                    vals.append(float(w))

    if len(vals) == 0:
        A = sp.coo_matrix((n, n), dtype=np.float32)
        print("  WARNING: all edge weights zero → empty graph")
    else:
        A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)

    # Symmetric normalize with self-loop: D^{-1/2}(A+I)D^{-1/2}
    A_csr = A.tocsr() + sp.eye(n, dtype=np.float32, format="csr")
    rowsum = np.array(A_csr.sum(1)).ravel()
    rowsum[rowsum == 0.0] = 1.0
    d_inv_sqrt = np.power(rowsum, -0.5)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_hat = (D_inv_sqrt @ A_csr @ D_inv_sqrt).tocoo()

    # Save
    np.savez_compressed(
        os.path.join(output_dir, "norm_adj_coo.npz"),
        row=A_hat.row, col=A_hat.col, data=A_hat.data, shape=A_hat.shape,
    )

    # Save address_to_index
    with open(os.path.join(output_dir, f"data_B4E.address_to_index"), "wb") as f:
        pickle.dump(address_to_index, f)

    print(f"  Adjacency: shape={A_hat.shape}, nnz={A_hat.nnz}")
    print(f"  address_to_index: {len(address_to_index)} accounts")

    return address_to_index


# ============================================================
# 9. BUILD TIMESERIES (aligned)
# ============================================================
def extract_timeseries(txs, max_seq_len=50):
    """10 features per timestep, pad/truncate."""
    if not txs:
        return np.zeros((max_seq_len, 10), dtype=np.float32)

    features = []
    cumul = 0.0
    for i, t in enumerate(txs):
        ts = int(t.get("timestamp", 0))
        amt = float(t.get("amount", 0.0))
        try:
            dt = datetime.fromtimestamp(ts)
            hour, dow = dt.hour, dt.weekday()
        except (ValueError, OSError, OverflowError):
            hour, dow = 0, 0

        td = float(ts - int(txs[i-1].get("timestamp", 0))) if i > 0 else 0.0
        cumul += amt
        features.append([
            amt, td, float(t.get("in_out", 0)),
            float(hour), float(dow), cumul,
            float(t.get("2-gram", 0.0)), float(t.get("3-gram", 0.0)),
            float(t.get("4-gram", 0.0)), float(t.get("5-gram", 0.0)),
        ])

    seq = np.array(features, dtype=np.float32)
    if len(seq) > max_seq_len:
        seq = seq[-max_seq_len:]
    if len(seq) < max_seq_len:
        seq = np.vstack([np.zeros((max_seq_len - len(seq), 10), dtype=np.float32), seq])
    return seq


def build_timeseries(accounts, alignment, max_seq_len=50):
    """Build timeseries theo thứ tự shuffled_clean_docs."""
    print("[9/10] Building aligned timeseries...")

    all_ts = []
    for addr in tqdm(alignment["final_addresses"], desc="Timeseries"):
        all_ts.append(extract_timeseries(accounts.get(addr, []), max_seq_len))

    X = np.array(all_ts, dtype=np.float32)

    # Normalize (log1p + min-max)
    print("  Normalizing...")
    X_n = X.copy()
    for c in [0, 1, 5, 6, 7, 8, 9]:
        X_n[:, :, c] = np.log1p(np.abs(X_n[:, :, c]))
    for c in range(10):
        mn, mx = X_n[:, :, c].min(), X_n[:, :, c].max()
        if mx - mn > 0:
            X_n[:, :, c] = (X_n[:, :, c] - mn) / (mx - mn)

    ts = alignment["train_size"]
    vs = alignment["valid_size"]
    X_tr = X_n[:ts]
    X_va = X_n[ts:ts + vs]
    X_te = X_n[ts + vs:]

    print(f"  X_train: {X_tr.shape}, X_valid: {X_va.shape}, X_test: {X_te.shape}")
    return X_tr, X_va, X_te


# ============================================================
# 10. SAVE ALL PREPROCESSED FILES
# ============================================================
def save_all(output_dir, ds_name, alignment, accounts, X_tr, X_va, X_te):
    """Save tất cả files cho ETH-GBert pipeline."""
    print("[10/10] Saving all preprocessed files...")

    final_tags = alignment["final_tags"]
    ts = alignment["train_size"]
    vs = alignment["valid_size"]

    y = np.array(final_tags, dtype=np.int64)
    train_y = y[:ts]
    valid_y = y[ts:ts + vs]
    test_y = y[ts + vs:]

    label2idx = {"0": 0, "1": 1}
    idx2label = {0: "0", 1: "1"}
    n_classes = len(label2idx)

    y_prob = np.eye(len(y), n_classes)[y]
    train_y_prob = y_prob[:ts]
    valid_y_prob = y_prob[ts:ts + vs]
    test_y_prob = y_prob[ts + vs:]

    # Tạo shuffled_clean_docs (text theo thứ tự cuối cùng)
    shuffled_clean_docs = []
    for addr in alignment["final_addresses"]:
        txs = accounts.get(addr, [])
        _, sentence = account_to_sentence(txs)
        shuffled_clean_docs.append(sentence)

    # Save pickle files
    prefix = f"data_{ds_name}"
    saves = {
        f"{prefix}.labels": [label2idx, idx2label],
        f"{prefix}.y": y,
        f"{prefix}.y_prob": y_prob,
        f"{prefix}.train_y": train_y,
        f"{prefix}.train_y_prob": train_y_prob,
        f"{prefix}.valid_y": valid_y,
        f"{prefix}.valid_y_prob": valid_y_prob,
        f"{prefix}.test_y": test_y,
        f"{prefix}.test_y_prob": test_y_prob,
        f"{prefix}.shuffled_clean_docs": shuffled_clean_docs,
    }

    for name, obj in saves.items():
        path = os.path.join(output_dir, name)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved: {name}")

    # Timeseries
    np.save(os.path.join(output_dir, "X_timeseries_train.npy"), X_tr)
    np.save(os.path.join(output_dir, "X_timeseries_valid.npy"), X_va)
    np.save(os.path.join(output_dir, "X_timeseries_test.npy"), X_te)
    print(f"  Saved: X_timeseries_train/valid/test.npy")

    # Alignment mapping
    with open(os.path.join(output_dir, "address_alignment.pkl"), "wb") as f:
        pickle.dump(alignment, f)
    print(f"  Saved: address_alignment.pkl")


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    start = time.time()
    print("=" * 60)
    print("  PREPROCESS B4E FOR ETH-GBert")
    print(f"  Graph:  {args.graph_file}")
    print(f"  Output: {args.output_dir}")
    print(f"  Ratio:  {args.ratio}:1 | Seed: {args.seed}")
    print("=" * 60 + "\n")

    # Load graph
    print("Loading graph...")
    with open(args.graph_file, "rb") as f:
        G = pickle.load(f, encoding="latin1")
    print(f"  Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}\n")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1-2: Extract + n-grams
    accounts = extract_transactions(G)
    accounts = add_time_ngrams(accounts)

    # 3: Balance
    selected = balance_accounts(accounts, args.ratio, args.seed)

    # 4: Generate text
    print("[4/10] Generating text sentences...")
    tags, sentences, addresses = [], [], []
    for addr in selected:
        tag, sentence = account_to_sentence(accounts[addr])
        tags.append(tag)
        sentences.append(sentence)
        addresses.append(addr)
    print(f"  Generated {len(sentences)} sentences")

    # 5: Split
    idx_train, idx_valid, idx_test = split_dataset(addresses, tags, args.seed)

    # 6: Save TSVs
    save_tsvs(tags, sentences, idx_train, idx_valid, idx_test, args.output_dir)

    # 7: Replay shuffle
    alignment = replay_bert_shuffle(addresses, tags, idx_train, idx_valid, args.seed)

    # 8: Adjacency matrix
    address_to_index = build_adjacency(accounts, selected, args.output_dir, args.seed)

    # 9: Timeseries
    X_tr, X_va, X_te = build_timeseries(accounts, alignment, args.max_seq_len)

    # 10: Save all
    save_all(args.output_dir, args.dataset_name, alignment, accounts,
             X_tr, X_va, X_te)

    # Summary
    elapsed = (time.time() - start) / 60
    total = alignment["train_size"] + alignment["valid_size"] + alignment["test_size"]
    y = np.array(alignment["final_tags"])

    print(f"\n{'='*60}")
    print(f"  PREPROCESSING COMPLETE! ({elapsed:.1f} min)")
    print(f"{'='*60}")
    print(f"  Total accounts:    {total:,}")
    print(f"  Phishing:          {y.sum():,} ({y.sum()/len(y)*100:.1f}%)")
    print(f"  Normal:            {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"  Train:             {alignment['train_size']:,}")
    print(f"  Valid:             {alignment['valid_size']:,}")
    print(f"  Test:              {alignment['test_size']:,}")
    print(f"  Adjacency:         {len(address_to_index)} nodes")
    print(f"  Timeseries:        ({total}, {args.max_seq_len}, 10)")
    print(f"")
    print(f"  TIẾP THEO (chạy từ ~/Dynamic_Feature_B4E/):")
    print(f"  1) Copy TSVs ra project root cho BERT_text_data.py:")
    print(f"     cp {args.output_dir}/train.tsv ~/Dynamic_Feature_B4E/")
    print(f"     cp {args.output_dir}/dev.tsv ~/Dynamic_Feature_B4E/")
    print(f"  2) Chạy BERT_text_data.py từ project root:")
    print(f"     cd ~/Dynamic_Feature_B4E")
    print(f"     python Dataset/BERT_text_data.py --ds B4E")
    print(f"     mv data_B4E.* {args.output_dir}/")
    print(f"  3) MLM pretrain:")
    print(f"     python make_mlm_corpus.py --data-dir {args.output_dir} --output-file mlm_corpus_b4e.txt --include-test")
    print(f"     python train_domain_bert_mlm.py --train-file mlm_corpus_b4e.txt --output-dir ./mlm_output_b4e/")
    print(f"  4) Train ETH-GBert:")
    print(f"     python train.py --ds B4E --bert_dir ./mlm_output_b4e/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()