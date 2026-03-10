import os
import re
import math
import pickle
import random
import pandas as pd
from collections import defaultdict

IN_PKL = os.path.join(os.path.dirname(__file__), "transactions4.pkl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed", "B4E")

random.seed(42)

def fmt_amt(x: float) -> str:
    # bucket rất thô để sentence không quá đa dạng; bạn có thể đổi sau
    if x <= 0: return "amt0"
    lg = math.log10(x)
    b = int(lg)  # order of magnitude
    return f"amt1e{b}"

def clean_addr(a: str) -> str:
    a = str(a).lower()
    return a

def bucket_dt(seconds: int) -> str:
    s = int(seconds)
    if s <= 0:
        return "eq0"
    if s < 60:
        return "lt1m"
    if s < 10*60:
        return "lt10m"
    if s < 60*60:
        return "lt1h"
    if s < 24*60*60:
        return "lt1d"
    if s < 7*24*60*60:
        return "lt7d"
    return "ge7d"

def build_sentence(txs):
    # map io + amount bucket + dt bins -> [unusedX] tokens (đảm bảo nằm trong vocab)
    DT_BINS = ["eq0","lt1m","lt10m","lt1h","lt1d","lt7d","ge7d"]
    TOK_IN  = "[unused0]"
    TOK_OUT = "[unused1]"

    # amount: amt0 -> unused9, amt1e0..amt1e29 -> unused10..unused39
    TOK_AMT0 = "[unused9]"
    AMT_BASE = 10
    AMT_MAX_EXP = 29

    # dt bins: 7 bins, mỗi n-gram dùng 7 unused tokens liên tiếp
    DT_BASE = {2: 40, 3: 47, 4: 54, 5: 61}

    def tok_amt(amount: float) -> str:
        if amount <= 0:
            return TOK_AMT0
        exp = int(math.log10(amount))
        exp = min(max(exp, 0), AMT_MAX_EXP)
        return f"[unused{AMT_BASE + exp}]"

    def tok_dt(n: int, b: str) -> str:
        j = DT_BINS.index(b) if b in DT_BINS else 0
        return f"[unused{DT_BASE[n] + j}]"

    toks = []
    for tx in txs:
        io_tok = TOK_IN if int(tx.get("in_out",0)) == 0 else TOK_OUT
        amt_tok = tok_amt(float(tx.get("amount",0.0)))

        d2 = bucket_dt(tx.get("dt_2gram",0))
        d3 = bucket_dt(tx.get("dt_3gram",0))
        d4 = bucket_dt(tx.get("dt_4gram",0))
        d5 = bucket_dt(tx.get("dt_5gram",0))

        toks.append(f"{io_tok} {amt_tok} {tok_dt(2,d2)} {tok_dt(3,d3)} {tok_dt(4,d4)} {tok_dt(5,d5)}")
    return " ".join(toks)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = pickle.load(open(IN_PKL, "rb"))

    rows = []
    for acc, txs in data.items():
        if not txs: 
            continue
        label = int(txs[0].get("tag", 0))
        sent = build_sentence(txs)
        rows.append((label, sent))

    df = pd.DataFrame(rows, columns=["label","sentence"])
    print("[OK] total rows:", len(df), "label counts:", df["label"].value_counts().to_dict())

    # split: 80/10/10
    idx = list(range(len(df)))
    random.shuffle(idx)
    n = len(idx)
    n_train = int(0.8*n)
    n_dev   = int(0.1*n)
    train_idx = idx[:n_train]
    dev_idx   = idx[n_train:n_train+n_dev]
    test_idx  = idx[n_train+n_dev:]

    df.iloc[train_idx].to_csv(os.path.join(OUT_DIR,"train.tsv"), sep="\t", index=False)
    df.iloc[dev_idx].to_csv(os.path.join(OUT_DIR,"dev.tsv"), sep="\t", index=False)
    df.iloc[test_idx].to_csv(os.path.join(OUT_DIR,"test.tsv"), sep="\t", index=False)

    print("[SAVE]", OUT_DIR)
    print("train/dev/test:", len(train_idx), len(dev_idx), len(test_idx))

if __name__ == "__main__":
    main()
