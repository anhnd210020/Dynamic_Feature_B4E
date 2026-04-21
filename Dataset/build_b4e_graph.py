"""
build_b4e_graph.py

Bước 1: Tạo MulDiGraph.pkl từ 4 file CSV của B4E dataset.

Input:
    data/raw/B4E/normal_trans/normal_eoa_transaction_in_slice_1000K.csv
    data/raw/B4E/normal_trans/normal_eoa_transaction_out_slice_1000K.csv
    data/raw/B4E/phish_trans/phisher_transaction_in.csv
    data/raw/B4E/phish_trans/phisher_transaction_out.csv

Output:
    Dataset/MulDiGraph.pkl

Cách chạy:
    cd ~/Dynamic_Feature_B4E/Dataset
    python build_b4e_graph.py
"""

import os
import pickle
import time

import networkx as nx
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Dataset/
PROJECT_DIR = os.path.dirname(BASE_DIR)                # Dynamic_Feature_B4E/
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw", "B4E")
OUTPUT_FILE = os.path.join(BASE_DIR, "MulDiGraph.pkl")

COLUMNS = [
    "tx_hash", "nonce", "block_hash", "block_number", "tx_index",
    "from_address", "to_address", "value", "gas", "gas_price",
    "input_data", "timestamp", "col12", "col13", "col14",
]

FILES = {
    "normal_in": {
        "path": os.path.join(RAW_DIR, "normal_trans", "normal_eoa_transaction_in_slice_1000K.csv"),
        "target_col": "to_address",      # account nhận = normal
        "label": 0,
    },
    "normal_out": {
        "path": os.path.join(RAW_DIR, "normal_trans", "normal_eoa_transaction_out_slice_1000K.csv"),
        "target_col": "from_address",    # account gửi = normal
        "label": 0,
    },
    "phish_in": {
        "path": os.path.join(RAW_DIR, "phish_trans", "phisher_transaction_in.csv"),
        "target_col": "to_address",      # account nhận = phishing
        "label": 1,
    },
    "phish_out": {
        "path": os.path.join(RAW_DIR, "phish_trans", "phisher_transaction_out.csv"),
        "target_col": "from_address",    # account gửi = phishing
        "label": 1,
    },
}


# ============================================================
# 1. IDENTIFY LABELED ACCOUNTS
# ============================================================
def identify_labeled_accounts():
    """
    Xác định tập normal accounts và phishing accounts.
    - normal = union(to_addr from normal_in, from_addr from normal_out)
    - phishing = union(to_addr from phish_in, from_addr from phish_out)
    """
    print("[1/4] Identifying labeled accounts...")

    account_labels = {}  # addr → label (0 or 1)

    for name, info in FILES.items():
        print(f"  Reading {name}...")
        df = pd.read_csv(info["path"], header=None, names=COLUMNS,
                         usecols=[COLUMNS.index(info["target_col"])],
                         low_memory=False, on_bad_lines="skip")

        addrs = df[info["target_col"]].dropna().unique()
        label = info["label"]

        for addr in addrs:
            if addr in account_labels:
                # Phishing label wins if conflict (shouldn't happen, overlap=0)
                if label == 1:
                    account_labels[addr] = 1
            else:
                account_labels[addr] = label

    n_normal = sum(1 for v in account_labels.values() if v == 0)
    n_phish = sum(1 for v in account_labels.values() if v == 1)
    print(f"  Labeled accounts: normal={n_normal:,}, phishing={n_phish:,}, "
          f"total={len(account_labels):,}")

    return account_labels


# ============================================================
# 2. BUILD GRAPH
# ============================================================
def build_graph(account_labels):
    """
    Tạo NetworkX MultiDiGraph từ tất cả transactions.
    - Nodes có attribute 'isp' (0=normal, 1=phishing, -1=unknown)
    - Edges có attributes 'amount', 'timestamp'
    """
    print("\n[2/4] Building MultiDiGraph...")

    G = nx.MultiDiGraph()
    total_edges = 0
    dedup_set = set()

    for name, info in FILES.items():
        print(f"  Processing {name}...")
        path = info["path"]

        # Đọc theo chunks để tiết kiệm RAM
        chunk_iter = pd.read_csv(
            path, header=None, names=COLUMNS,
            usecols=["from_address", "to_address", "value", "timestamp"],
            low_memory=False, on_bad_lines="skip",
            chunksize=500_000,
        )

        for chunk in tqdm(chunk_iter, desc=f"  {name}"):
            for _, row in chunk.iterrows():
                fa = row["from_address"]
                ta = row["to_address"]

                if pd.isna(fa) or pd.isna(ta):
                    continue

                fa = str(fa).strip()
                ta = str(ta).strip()

                if not fa or not ta:
                    continue

                # Parse value (wei → float)
                try:
                    amount = float(row["value"])
                except (ValueError, TypeError):
                    amount = 0.0

                # Parse timestamp
                try:
                    ts = int(row["timestamp"])
                except (ValueError, TypeError):
                    ts = 0

                # Dedup: same (from, to, amount, timestamp) = same edge
                edge_key = (fa, ta, amount, ts)
                if edge_key in dedup_set:
                    continue
                dedup_set.add(edge_key)

                # Add nodes with labels
                if fa not in G:
                    G.add_node(fa, isp=account_labels.get(fa, -1))
                if ta not in G:
                    G.add_node(ta, isp=account_labels.get(ta, -1))

                # Add edge
                G.add_edge(fa, ta, amount=amount, timestamp=ts)
                total_edges += 1

    print(f"  Graph built: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")

    # Stats on labeled nodes
    labeled_nodes = {n: d.get("isp", -1) for n, d in G.nodes(data=True)}
    n0 = sum(1 for v in labeled_nodes.values() if v == 0)
    n1 = sum(1 for v in labeled_nodes.values() if v == 1)
    n_unk = sum(1 for v in labeled_nodes.values() if v == -1)
    print(f"  Node labels: normal={n0:,}, phishing={n1:,}, unknown={n_unk:,}")

    return G


# ============================================================
# 3. CLEAN GRAPH
# ============================================================
def clean_graph(G):
    """
    Xóa nodes không có label (isp=-1) và chỉ giữ nodes
    có ít nhất 1 transaction (degree > 0).
    Giữ nodes unknown nếu chúng là counterparty của labeled nodes.
    """
    print("\n[3/4] Cleaning graph...")

    # Giữ tất cả nodes (kể cả unknown counterparties)
    # Chỉ xóa isolated nodes (degree=0)
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)
    print(f"  Removed {len(isolated):,} isolated nodes")
    print(f"  After clean: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")

    return G


# ============================================================
# 4. SAVE
# ============================================================
def save_graph(G):
    print(f"\n[4/4] Saving graph to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(G, f)

    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"  Saved: {file_size:.1f} MB")


# ============================================================
# MAIN
# ============================================================
def main():
    start = time.time()
    print("=" * 60)
    print("  BUILD B4E MULDIGRAPH")
    print("=" * 60 + "\n")

    # Check files
    for name, info in FILES.items():
        exists = "✓" if os.path.exists(info["path"]) else "✗"
        print(f"  {exists} {name}: {info['path']}")
    print()

    # Pipeline
    account_labels = identify_labeled_accounts()
    G = build_graph(account_labels)
    G = clean_graph(G)
    save_graph(G)

    elapsed = (time.time() - start) / 60
    print(f"\n{'='*60}")
    print(f"  DONE! ({elapsed:.1f} minutes)")
    print(f"{'='*60}")
    print(f"  Nodes:       {G.number_of_nodes():,}")
    print(f"  Edges:       {G.number_of_edges():,}")

    labeled = {n: d["isp"] for n, d in G.nodes(data=True) if d.get("isp", -1) != -1}
    n0 = sum(1 for v in labeled.values() if v == 0)
    n1 = sum(1 for v in labeled.values() if v == 1)
    print(f"  Normal:      {n0:,}")
    print(f"  Phishing:    {n1:,}")
    print(f"  Output:      {OUTPUT_FILE}")
    print(f"\n  TIẾP THEO:")
    print(f"  python preprocess_b4e.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()