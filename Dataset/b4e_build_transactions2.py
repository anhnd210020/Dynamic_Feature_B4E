import os
import pickle
import argparse
from collections import defaultdict

import pandas as pd


def iter_transactions_csv(path: str, label: int, in_out: int, account_side: str, chunksize: int):
    """
    Read a B4E_fixed CSV in chunks and yield (account, tx_dict).
    account_side:
      - "to"   : account = to_address (for *_in.csv)
      - "from" : account = from_address (for *_out.csv)
    """
    usecols = ["from_address", "to_address", "value", "timestamp"]

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype=str):
        # coerce types safely (value can be huge -> keep float for downstream robustness)
        chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce").fillna(0).astype("int64")
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce").fillna(0.0).astype("float64")

        if account_side == "to":
            acc_series = chunk["to_address"].astype(str)
        elif account_side == "from":
            acc_series = chunk["from_address"].astype(str)
        else:
            raise ValueError("account_side must be 'to' or 'from'")

        from_s = chunk["from_address"].astype(str)
        to_s   = chunk["to_address"].astype(str)
        val_s  = chunk["value"]
        ts_s   = chunk["timestamp"]

        # fast row iteration with itertuples over numpy arrays
        for acc, f, t, v, ts in zip(acc_series.values, from_s.values, to_s.values, val_s.values, ts_s.values):
            if not acc or acc == "nan":
                continue
            tx = {
                "tag": int(label),        # account label
                "from_address": str(f),
                "to_address": str(t),
                "amount": float(v),       # store float to avoid int overflow
                "timestamp": int(ts),
                "in_out": int(in_out),    # 0=in, 1=out
            }
            yield str(acc), tx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default="data/raw/B4E_fixed", help="Root folder containing phish_trans/ and normal_trans/")
    ap.add_argument("--chunksize", type=int, default=300_000, help="CSV chunk size")
    ap.add_argument("--skip_sort", action="store_true", help="Skip per-account timestamp sorting (NOT recommended)")
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_root = os.path.join(root, args.raw_root)

    # Expected filenames (from your current download)
    PHISH_IN  = os.path.join(raw_root, "phish_trans",  "phisher_transaction_in.csv")
    PHISH_OUT = os.path.join(raw_root, "phish_trans",  "phisher_transaction_out.csv")
    NORM_IN   = os.path.join(raw_root, "normal_trans", "normal_eoa_transaction_in_slice_1000K.csv")
    NORM_OUT  = os.path.join(raw_root, "normal_trans", "normal_eoa_transaction_out_slice_1000K.csv")

    for p in [PHISH_IN, PHISH_OUT, NORM_IN, NORM_OUT]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    out_pkl = os.path.join(os.path.dirname(__file__), "transactions2.pkl")

    accounts = defaultdict(list)

    # phish label=1
    print("[READ]", PHISH_IN)
    for acc, tx in iter_transactions_csv(PHISH_IN, label=1, in_out=0, account_side="to", chunksize=args.chunksize):
        accounts[acc].append(tx)

    print("[READ]", PHISH_OUT)
    for acc, tx in iter_transactions_csv(PHISH_OUT, label=1, in_out=1, account_side="from", chunksize=args.chunksize):
        accounts[acc].append(tx)

    # normal label=0
    print("[READ]", NORM_IN)
    for acc, tx in iter_transactions_csv(NORM_IN, label=0, in_out=0, account_side="to", chunksize=args.chunksize):
        accounts[acc].append(tx)

    print("[READ]", NORM_OUT)
    for acc, tx in iter_transactions_csv(NORM_OUT, label=0, in_out=1, account_side="from", chunksize=args.chunksize):
        accounts[acc].append(tx)

    print(f"[OK] accounts={len(accounts)} (before sort)")

    if not args.skip_sort:
        # sort each account's tx list by timestamp
        for acc in accounts:
            accounts[acc].sort(key=lambda t: int(t.get("timestamp", 0)))
        print("[OK] sorted by timestamp")

    # quick preview
    shown = 0
    for acc, txs in accounts.items():
        if not txs:
            continue
        print("Account:", acc, "| txs:", len(txs), "| tag:", txs[0]["tag"], "| first:", txs[0])
        shown += 1
        if shown >= 3:
            break

    with open(out_pkl, "wb") as f:
        pickle.dump(dict(accounts), f)

    print("[SAVE]", out_pkl)


if __name__ == "__main__":
    main()
