import os
import glob
import pandas as pd

# Cột chuẩn theo thứ tự (dựa trên format bạn đã in ra)
BASE_COLS = [
    "tx_hash", "nonce", "block_hash", "block_number", "tx_index",
    "from_address", "to_address", "value", "gas", "gas_price", "input", "timestamp"
]

def read_noheader_csv(path: str) -> pd.DataFrame:
    # Đọc file KHÔNG header
    df = pd.read_csv(path, header=None, dtype=str)

    if df.shape[1] < len(BASE_COLS):
        raise ValueError(f"{path}: expected >= {len(BASE_COLS)} columns, got {df.shape[1]}")

    # Chỉ lấy 12 cột đầu (phần còn lại thường là extra/empty)
    df = df.iloc[:, :len(BASE_COLS)].copy()
    df.columns = BASE_COLS

    # Clean nhẹ
    df["from_address"] = df["from_address"].astype(str)
    df["to_address"]   = df["to_address"].astype(str)

    # ép kiểu số cho value/timestamp (để downstream dễ)
    df["value"] = df["value"].fillna("0").astype(str)  # giữ string để không overflow
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype("int64")

    # bỏ dòng thiếu from/to
    df = df.dropna(subset=["from_address", "to_address"])
    df = df[(df["from_address"].str.len() > 0) & (df["to_address"].str.len() > 0)]

    return df

def mirror_out_path(in_path: str, in_root: str, out_root: str) -> str:
    rel = os.path.relpath(in_path, in_root)
    return os.path.join(out_root, rel)

def main():
    in_root  = os.path.join("data", "raw", "B4E")
    out_root = os.path.join("data", "raw", "B4E_fixed")

    os.makedirs(out_root, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_root, "*", "*.csv")))
    print("Found CSV files:", len(files))
    if not files:
        print("No CSV found under", in_root)
        return

    for path in files:
        df = read_noheader_csv(path)
        out_path = mirror_out_path(path, in_root, out_root)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        df.to_csv(out_path, index=False)
        print("[OK]", path, "->", out_path, "rows:", len(df))

    # quick check: show head of one file
    sample = mirror_out_path(files[0], in_root, out_root)
    print("\nSample preview:", sample)
    print(pd.read_csv(sample, nrows=3).to_string(index=False))

if __name__ == "__main__":
    main()
