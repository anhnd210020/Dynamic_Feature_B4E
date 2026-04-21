"""
analyze_b4e_dataset.py

Phân tích dataset B4E: normal_trans + phish_trans.
Chạy: python analyze_b4e_dataset.py
"""

import os
import pandas as pd
import numpy as np
from collections import Counter

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "B4E")

FILES = {
    "normal_in": os.path.join(RAW_DIR, "normal_trans", "normal_eoa_transaction_in_slice_1000K.csv"),
    "normal_out": os.path.join(RAW_DIR, "normal_trans", "normal_eoa_transaction_out_slice_1000K.csv"),
    "phish_in": os.path.join(RAW_DIR, "phish_trans", "phisher_transaction_in.csv"),
    "phish_out": os.path.join(RAW_DIR, "phish_trans", "phisher_transaction_out.csv"),
}

# Columns (Ethereum raw transaction — no header)
COLUMNS = [
    "tx_hash",            # 0
    "nonce",              # 1
    "block_hash",         # 2
    "block_number",       # 3
    "tx_index",           # 4
    "from_address",       # 5
    "to_address",         # 6
    "value",              # 7  (wei)
    "gas",                # 8
    "gas_price",          # 9
    "input_data",         # 10
    "timestamp",          # 11
    "col12",              # 12 (trống?)
    "col13",              # 13 (trống?)
    "col14",              # 14 (trống?)
]


# ============================================================
# 1. LOAD + BASIC STATS
# ============================================================
def load_csv(path, name, nrows=None):
    print(f"\n  Loading {name}...")
    df = pd.read_csv(
        path, header=None, names=COLUMNS,
        low_memory=False, nrows=nrows,
        on_bad_lines="skip",
    )
    print(f"    Rows: {len(df):,}")
    print(f"    Columns: {len(df.columns)}")
    return df


def basic_stats(df, name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape}")
    print(f"\n  Dtypes:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = non_null / len(df) * 100
        print(f"    {col:20s}: {str(df[col].dtype):10s} | non-null: {non_null:>10,} ({pct:.1f}%)")

    # Sample rows
    print(f"\n  Sample (first 3 rows):")
    print(df.head(3).to_string(index=False))


# ============================================================
# 2. ADDRESS ANALYSIS
# ============================================================
def address_analysis(df_in, df_out, label_name):
    """Phân tích addresses cho 1 loại (normal hoặc phish)."""
    print(f"\n{'='*60}")
    print(f"  ADDRESS ANALYSIS: {label_name}")
    print(f"{'='*60}")

    # In transactions: to_address là target account
    in_addrs = set(df_in["to_address"].dropna().unique())
    # Out transactions: from_address là target account
    out_addrs = set(df_out["from_address"].dropna().unique())

    all_addrs = in_addrs | out_addrs
    both_addrs = in_addrs & out_addrs

    print(f"  Unique addresses (IN):   {len(in_addrs):,}")
    print(f"  Unique addresses (OUT):  {len(out_addrs):,}")
    print(f"  Unique addresses (ALL):  {len(all_addrs):,}")
    print(f"  Addresses in BOTH:       {len(both_addrs):,}")

    # Transactions per address
    in_counts = df_in["to_address"].value_counts()
    out_counts = df_out["from_address"].value_counts()

    print(f"\n  Transactions per address (IN):")
    print(f"    Mean:   {in_counts.mean():.1f}")
    print(f"    Median: {in_counts.median():.1f}")
    print(f"    Min:    {in_counts.min()}")
    print(f"    Max:    {in_counts.max():,}")

    print(f"\n  Transactions per address (OUT):")
    print(f"    Mean:   {out_counts.mean():.1f}")
    print(f"    Median: {out_counts.median():.1f}")
    print(f"    Min:    {out_counts.min()}")
    print(f"    Max:    {out_counts.max():,}")

    return all_addrs


# ============================================================
# 3. VALUE ANALYSIS
# ============================================================
def value_analysis(df, name):
    print(f"\n{'='*60}")
    print(f"  VALUE ANALYSIS: {name}")
    print(f"{'='*60}")

    # Convert value to float (wei → ETH)
    df_val = pd.to_numeric(df["value"], errors="coerce")
    valid = df_val.dropna()
    eth_values = valid / 1e18  # wei → ETH

    print(f"  Total transactions: {len(df):,}")
    print(f"  Valid value count:  {len(valid):,}")
    print(f"  Zero-value txs:    {(valid == 0).sum():,} ({(valid == 0).sum()/len(valid)*100:.1f}%)")

    non_zero = eth_values[eth_values > 0]
    if len(non_zero) > 0:
        print(f"\n  Value (ETH) — non-zero only:")
        print(f"    Mean:   {non_zero.mean():.4f}")
        print(f"    Median: {non_zero.median():.4f}")
        print(f"    Min:    {non_zero.min():.8f}")
        print(f"    Max:    {non_zero.max():,.2f}")
        print(f"    Std:    {non_zero.std():.4f}")

        # Distribution
        print(f"\n  Value distribution:")
        bins = [0, 0.01, 0.1, 1, 10, 100, 1000, float("inf")]
        labels = ["<0.01", "0.01-0.1", "0.1-1", "1-10", "10-100", "100-1K", ">1K"]
        cuts = pd.cut(non_zero, bins=bins, labels=labels)
        for label, count in cuts.value_counts().sort_index().items():
            pct = count / len(non_zero) * 100
            print(f"    {label:10s}: {count:>10,} ({pct:.1f}%)")


# ============================================================
# 4. TIMESTAMP ANALYSIS
# ============================================================
def timestamp_analysis(df, name):
    print(f"\n{'='*60}")
    print(f"  TIMESTAMP ANALYSIS: {name}")
    print(f"{'='*60}")

    ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
    if len(ts) == 0:
        print("  No valid timestamps")
        return

    dates = pd.to_datetime(ts, unit="s", errors="coerce")
    valid_dates = dates.dropna()

    print(f"  Valid timestamps: {len(valid_dates):,}")
    print(f"  Earliest:  {valid_dates.min()}")
    print(f"  Latest:    {valid_dates.max()}")
    print(f"  Span:      {(valid_dates.max() - valid_dates.min()).days} days")

    # Yearly distribution
    years = valid_dates.dt.year
    print(f"\n  Transactions per year:")
    for year, count in years.value_counts().sort_index().items():
        pct = count / len(valid_dates) * 100
        print(f"    {year}: {count:>10,} ({pct:.1f}%)")


# ============================================================
# 5. GAS ANALYSIS
# ============================================================
def gas_analysis(df, name):
    print(f"\n{'='*60}")
    print(f"  GAS ANALYSIS: {name}")
    print(f"{'='*60}")

    gas = pd.to_numeric(df["gas"], errors="coerce").dropna()
    gas_price = pd.to_numeric(df["gas_price"], errors="coerce").dropna()

    print(f"  Gas limit:")
    print(f"    Mean:   {gas.mean():,.0f}")
    print(f"    Median: {gas.median():,.0f}")
    print(f"    Min:    {gas.min():,.0f}")
    print(f"    Max:    {gas.max():,.0f}")

    if len(gas_price) > 0:
        gwei = gas_price / 1e9
        print(f"\n  Gas price (Gwei):")
        print(f"    Mean:   {gwei.mean():.2f}")
        print(f"    Median: {gwei.median():.2f}")


# ============================================================
# 6. INPUT DATA ANALYSIS
# ============================================================
def input_data_analysis(df, name):
    print(f"\n{'='*60}")
    print(f"  INPUT DATA ANALYSIS: {name}")
    print(f"{'='*60}")

    inputs = df["input_data"].fillna("").astype(str)
    simple_tx = (inputs == "0x").sum()
    contract_tx = (inputs.str.len() > 2).sum() & (inputs != "0x").sum()

    print(f"  Simple transfers (input='0x'):  {simple_tx:,} ({simple_tx/len(df)*100:.1f}%)")
    print(f"  Contract interactions:          {contract_tx:,} ({contract_tx/len(df)*100:.1f}%)")

    # Top function signatures (first 10 chars of input)
    contract_inputs = inputs[(inputs != "0x") & (inputs.str.len() > 10)]
    if len(contract_inputs) > 0:
        sigs = contract_inputs.str[:10]
        top_sigs = sigs.value_counts().head(10)
        print(f"\n  Top 10 function signatures:")
        for sig, count in top_sigs.items():
            print(f"    {sig}: {count:,}")


# ============================================================
# 7. OVERLAP ANALYSIS
# ============================================================
def overlap_analysis(normal_addrs, phish_addrs):
    print(f"\n{'='*60}")
    print(f"  OVERLAP ANALYSIS: Normal vs Phishing")
    print(f"{'='*60}")

    overlap = normal_addrs & phish_addrs
    only_normal = normal_addrs - phish_addrs
    only_phish = phish_addrs - normal_addrs

    print(f"  Normal addresses:    {len(normal_addrs):,}")
    print(f"  Phishing addresses:  {len(phish_addrs):,}")
    print(f"  Overlap:             {len(overlap):,}")
    print(f"  Only normal:         {len(only_normal):,}")
    print(f"  Only phishing:       {len(only_phish):,}")

    if len(overlap) > 0:
        print(f"\n  WARNING: {len(overlap)} addresses appear in BOTH normal and phishing!")
        print(f"  Sample overlap addresses: {list(overlap)[:5]}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  B4E DATASET ANALYSIS")
    print("=" * 60)

    # Check files
    for name, path in FILES.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {name}: {path}")

    # Load (dùng tất cả rows)
    print("\n[1/7] Loading data...")
    dfs = {}
    for name, path in FILES.items():
        dfs[name] = load_csv(path, name)

    # Basic stats
    print("\n[2/7] Basic statistics...")
    for name, df in dfs.items():
        basic_stats(df, name)

    # Address analysis
    print("\n[3/7] Address analysis...")
    normal_addrs = address_analysis(
        dfs["normal_in"], dfs["normal_out"], "NORMAL"
    )
    phish_addrs = address_analysis(
        dfs["phish_in"], dfs["phish_out"], "PHISHING"
    )

    # Value analysis
    print("\n[4/7] Value analysis...")
    for name, df in dfs.items():
        value_analysis(df, name)

    # Timestamp analysis
    print("\n[5/7] Timestamp analysis...")
    for name, df in dfs.items():
        timestamp_analysis(df, name)

    # Gas analysis
    print("\n[6/7] Gas analysis...")
    for name, df in dfs.items():
        gas_analysis(df, name)

    # Input data analysis + overlap
    print("\n[7/7] Input data & overlap...")
    for name, df in dfs.items():
        input_data_analysis(df, name)
    overlap_analysis(normal_addrs, phish_addrs)

    # Summary
    print("\n" + "#" * 60)
    print("  SUMMARY")
    print("#" * 60)
    total_txs = sum(len(df) for df in dfs.values())
    print(f"  Total transactions: {total_txs:,}")
    print(f"  Normal IN:          {len(dfs['normal_in']):,}")
    print(f"  Normal OUT:         {len(dfs['normal_out']):,}")
    print(f"  Phishing IN:        {len(dfs['phish_in']):,}")
    print(f"  Phishing OUT:       {len(dfs['phish_out']):,}")
    print(f"  Normal addresses:   {len(normal_addrs):,}")
    print(f"  Phishing addresses: {len(phish_addrs):,}")
    print(f"  Imbalance ratio:    1:{len(normal_addrs)//max(len(phish_addrs),1)}")
    print("#" * 60)


if __name__ == "__main__":
    main()