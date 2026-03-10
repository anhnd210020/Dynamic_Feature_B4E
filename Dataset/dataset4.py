import os
import pickle

IN_PKL  = os.path.join(os.path.dirname(__file__), "transactions3.pkl")
OUT_PKL = os.path.join(os.path.dirname(__file__), "transactions4.pkl")

def main():
    data = pickle.load(open(IN_PKL, "rb"))
    out = {}

    for acc, txs in data.items():
        if not txs:
            continue
        # đảm bảo sort theo timestamp
        txs = sorted(txs, key=lambda t: int(t.get("timestamp", 0)))

        # tiện: list timestamp
        ts = [int(t.get("timestamp", 0)) for t in txs]

        new_list = []
        for i, tx in enumerate(txs):
            tx2 = dict(tx)
            # n-gram time diff: ΔT_n = t_i - t_{i-(n-1)}
            # (n=2..5)
            for n in (2, 3, 4, 5):
                j = i - (n - 1)
                if j >= 0:
                    dt = max(0, ts[i] - ts[j])
                else:
                    dt = 0
                tx2[f"dt_{n}gram"] = int(dt)
            new_list.append(tx2)

        out[acc] = new_list

    pickle.dump(out, open(OUT_PKL, "wb"))
    print("[OK] wrote", OUT_PKL, "accounts:", len(out))

    # preview
    k = next(iter(out))
    print("sample:", k, "txs:", len(out[k]))
    print("first tx keys:", list(out[k][0].keys()))
    print("first 3 dt_2gram:", [out[k][i]["dt_2gram"] for i in range(min(3, len(out[k])))])

if __name__ == "__main__":
    main()
