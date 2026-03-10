import os
import pickle

IN_PKL  = os.path.join(os.path.dirname(__file__), "transactions2.pkl")
OUT_PKL = os.path.join(os.path.dirname(__file__), "transactions3.pkl")

def main():
    data = pickle.load(open(IN_PKL, "rb"))  # {account: [tx,...]}
    out = {}

    for acc, txs in data.items():
        if not txs:
            continue
        # đảm bảo sort theo timestamp
        txs = sorted(txs, key=lambda t: int(t.get("timestamp", 0)))

        prev_t = None
        new_list = []
        for tx in txs:
            t = int(tx.get("timestamp", 0))
            if prev_t is None:
                dt = 0
            else:
                dt = max(0, t - prev_t)
            prev_t = t

            tx2 = dict(tx)
            tx2["time_diff"] = int(dt)
            new_list.append(tx2)

        out[acc] = new_list

    pickle.dump(out, open(OUT_PKL, "wb"))
    print("[OK] wrote", OUT_PKL, "accounts:", len(out))

    # preview
    k = next(iter(out))
    print("sample:", k, "txs:", len(out[k]))
    print("first 3 time_diff:", [out[k][i]["time_diff"] for i in range(min(3, len(out[k])))])

if __name__ == "__main__":
    main()
