import os
import sys
import math
import pickle
import shutil
import random
import numpy as np
import pandas as pd

random.seed(42)

SRC = "B4E"   # nguồn gốc

def dump(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def main():
    if len(sys.argv) != 2:
        print("Usage: python Dataset/make_b4e_balanced_ratio.py <neg_per_pos>")
        print("Example: python Dataset/make_b4e_balanced_ratio.py 5")
        sys.exit(1)

    neg_per_pos = int(sys.argv[1])
    dst = f"B4E_bal_1to{neg_per_pos}"

    src_dir = os.path.join("data", "preprocessed", SRC)
    dst_dir = os.path.join("data", "preprocessed", dst)
    os.makedirs(dst_dir, exist_ok=True)

    # load source tsv
    train = pd.read_csv(os.path.join(src_dir, "train.tsv"), sep="\t")
    dev   = pd.read_csv(os.path.join(src_dir, "dev.tsv"), sep="\t")
    test  = pd.read_csv(os.path.join(src_dir, "test.tsv"), sep="\t")

    pos = train[train["label"] == 1].copy()
    neg = train[train["label"] == 0].copy()

    n_pos = len(pos)
    n_neg_target = min(len(neg), n_pos * neg_per_pos)

    neg_sub = neg.sample(n=n_neg_target, random_state=42)
    train_bal = pd.concat([pos, neg_sub], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # save tsv
    train_bal.to_csv(os.path.join(dst_dir, "train.tsv"), sep="\t", index=False)
    dev.to_csv(os.path.join(dst_dir, "dev.tsv"), sep="\t", index=False)
    test.to_csv(os.path.join(dst_dir, "test.tsv"), sep="\t", index=False)

    print("[OK] train label counts:", train_bal["label"].value_counts().to_dict())
    print("[OK] dev label counts:", dev["label"].value_counts().to_dict())
    print("[OK] test label counts:", test["label"].value_counts().to_dict())

    # labels file in exact format trainModel.py expects
    label2idx = {'0': 0, '1': 1}
    idx2label = {0: '0', 1: '1'}
    dump([label2idx, idx2label], os.path.join(dst_dir, f"data_{dst}.labels"))

    # docs in exact order = train + dev + test
    docs = pd.concat(
        [train_bal["sentence"], dev["sentence"], test["sentence"]],
        ignore_index=True
    ).astype(str).tolist()
    dump(docs, os.path.join(dst_dir, f"data_{dst}.shuffled_clean_docs"))

    # y arrays
    y_train = train_bal["label"].astype(int).to_numpy()
    y_dev   = dev["label"].astype(int).to_numpy()
    y_test  = test["label"].astype(int).to_numpy()
    y_all   = np.hstack([y_train, y_dev, y_test])

    dump(y_train, os.path.join(dst_dir, f"data_{dst}.train_y"))
    dump(y_dev,   os.path.join(dst_dir, f"data_{dst}.valid_y"))
    dump(y_test,  os.path.join(dst_dir, f"data_{dst}.test_y"))
    dump(y_all,   os.path.join(dst_dir, f"data_{dst}.y"))

    # y_prob: must be (N,1) so np.vstack works in your trainModel.py
    p_train = np.ones((len(y_train), 1), dtype=np.float32)
    p_dev   = np.ones((len(y_dev),   1), dtype=np.float32)
    p_test  = np.ones((len(y_test),  1), dtype=np.float32)
    p_all   = np.ones((len(y_all),   1), dtype=np.float32)

    dump(p_train, os.path.join(dst_dir, f"data_{dst}.train_y_prob"))
    dump(p_dev,   os.path.join(dst_dir, f"data_{dst}.valid_y_prob"))
    dump(p_test,  os.path.join(dst_dir, f"data_{dst}.test_y_prob"))
    dump(p_all,   os.path.join(dst_dir, f"data_{dst}.y_prob"))

    # copy graph artifacts from source B4E for now
    for fname_src, fname_dst in [
        (f"data_{SRC}.address_to_index", f"data_{dst}.address_to_index"),
        ("norm_adj_coo.npz", "norm_adj_coo.npz"),
    ]:
        s = os.path.join(src_dir, fname_src)
        t = os.path.join(dst_dir, fname_dst)
        if os.path.exists(s):
            shutil.copy2(s, t)
            print("[OK] copied", s, "->", t)
        else:
            print("[WARN] missing", s)

    print("[DONE] created", dst_dir)
    print("train/dev/test sizes:", len(train_bal), len(dev), len(test))

if __name__ == "__main__":
    main()
