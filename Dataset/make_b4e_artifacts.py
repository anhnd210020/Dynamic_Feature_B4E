import os
import pickle
import numpy as np
import pandas as pd

DS = "B4E"
BASE = os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed", DS)

def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def main():
    train = pd.read_csv(os.path.join(BASE, "train.tsv"), sep="\t")
    dev   = pd.read_csv(os.path.join(BASE, "dev.tsv"), sep="\t")
    test  = pd.read_csv(os.path.join(BASE, "test.tsv"), sep="\t")

    # clean docs list (train+dev+test)
    docs = pd.concat([train["sentence"], dev["sentence"], test["sentence"]], ignore_index=True)\
             .astype(str).tolist()

    labels_all = pd.concat([train["label"], dev["label"], test["label"]], ignore_index=True)\
                   .astype(int).values

    train_y = train["label"].astype(int).values
    valid_y = dev["label"].astype(int).values
    test_y  = test["label"].astype(int).values

    # The project often stores y_prob as class distribution / priors (not model probs).
    # We'll store a simple prior vector [P(y=0), P(y=1)] for each split.
    def prior(y):
        c0 = float((y==0).sum()); c1 = float((y==1).sum())
        s = c0+c1
        return np.array([c0/s, c1/s], dtype=np.float32)

    train_y_prob = prior(train_y)
    valid_y_prob = prior(valid_y)
    test_y_prob  = prior(test_y)
    y_prob       = prior(labels_all)

    # Save in the same naming style as your multigraph run
    save_pkl(labels_all.tolist(), os.path.join(BASE, f"data_{DS}.labels"))
    save_pkl(docs,              os.path.join(BASE, f"data_{DS}.shuffled_clean_docs"))

    np.save(os.path.join(BASE, f"data_{DS}.train_y.npy"), train_y)
    np.save(os.path.join(BASE, f"data_{DS}.valid_y.npy"), valid_y)
    np.save(os.path.join(BASE, f"data_{DS}.test_y.npy"),  test_y)
    np.save(os.path.join(BASE, f"data_{DS}.y.npy"),       labels_all)

    np.save(os.path.join(BASE, f"data_{DS}.train_y_prob.npy"), train_y_prob)
    np.save(os.path.join(BASE, f"data_{DS}.valid_y_prob.npy"), valid_y_prob)
    np.save(os.path.join(BASE, f"data_{DS}.test_y_prob.npy"),  test_y_prob)
    np.save(os.path.join(BASE, f"data_{DS}.y_prob.npy"),       y_prob)

    print("[OK] wrote artifacts to", BASE)
    print("docs:", len(docs),
          "train/dev/test:", len(train_y), len(valid_y), len(test_y),
          "label_counts(all):", {0:int((labels_all==0).sum()), 1:int((labels_all==1).sum())})

if __name__ == "__main__":
    main()
