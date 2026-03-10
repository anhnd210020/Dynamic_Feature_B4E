import os, pickle
import numpy as np
import scipy.sparse as sp

DS = "B4E"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed", DS)

def main():
    os.makedirs(OUT, exist_ok=True)

    # vocab map: include unused tokens 0..99
    vocab = {f"[unused{i}]": i for i in range(100)}
    vocab["[PAD]"] = 100
    vocab["[CLS]"] = 101
    vocab["[SEP]"] = 102
    vocab["[MASK]"] = 103
    vocab["[UNK]"] = 104

    with open(os.path.join(OUT, f"data_{DS}.address_to_index"), "wb") as f:
        pickle.dump(vocab, f)

    # norm_adj_coo: identity adjacency (sparse)
    n = len(vocab)
    I = sp.identity(n, format="coo", dtype=np.float32)
    sp.save_npz(os.path.join(OUT, "norm_adj_coo.npz"), I)

    print("[OK] wrote", os.path.join(OUT, f"data_{DS}.address_to_index"))
    print("[OK] wrote", os.path.join(OUT, "norm_adj_coo.npz"), "shape:", I.shape, "nnz:", I.nnz)

if __name__ == "__main__":
    main()
