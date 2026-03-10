import os
from types import SimpleNamespace

# ===== Dataset paths =====
DATASET = "B4E"
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PREPROCESSED_DIR = os.path.join(ROOT_DIR, "data", "preprocessed", DATASET)
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_TSV = os.path.join(PREPROCESSED_DIR, "train.tsv")
DEV_TSV   = os.path.join(PREPROCESSED_DIR, "dev.tsv")
TEST_TSV  = os.path.join(PREPROCESSED_DIR, "test.tsv")

NORM_ADJ_PATH   = os.path.join(PREPROCESSED_DIR, "norm_adj_coo.npz")
VOCAB_MAP_PATH  = os.path.join(PREPROCESSED_DIR, f"data_{DATASET}.address_to_index")
LABELS_PATH     = os.path.join(PREPROCESSED_DIR, f"data_{DATASET}.labels")
DOCS_PATH       = os.path.join(PREPROCESSED_DIR, f"data_{DATASET}.shuffled_clean_docs")
TRAIN_Y_PATH    = os.path.join(PREPROCESSED_DIR, f"data_{DATASET}.train_y.npy")
VALID_Y_PATH    = os.path.join(PREPROCESSED_DIR, f"data_{DATASET}.valid_y.npy")
TEST_Y_PATH     = os.path.join(PREPROCESSED_DIR, f"data_{DATASET}.test_y.npy")
TRAIN_Y_PROB_PATH = os.path.join(PREPROCESSED_DIR, f"data_{DATASET}.train_y_prob.npy")
VALID_Y_PROB_PATH = os.path.join(PREPROCESSED_DIR, f"data_{DATASET}.valid_y_prob.npy")
TEST_Y_PROB_PATH  = os.path.join(PREPROCESSED_DIR, f"data_{DATASET}.test_y_prob.npy")

# ===== Training / model knobs (defaults) =====
GLOBAL_SEED = 42
BERT_NAME = "bert-base-uncased"

# Offline / local model cache controls (safe defaults)
TRANSFORMERS_OFFLINE = False
HUGGING_LOCAL_MODEL_FILES_PATH = ""  # set to a local folder if you have one

# Aux MLM
AUX_MLM = 1
MLM_PROB = 0.15
MLM_LAMBDA = 0.1

# A3 (you said A3=0 earlier, keep disabled)
A3_ENABLE = 0
A3_TOPK_ADDR = 50
A3_UNUSED_BUDGET = 30
A3_DT_BINS = 7
A3_VAL_BINS = 8

# ===== Export object expected by trainModel.py =====
env_config = SimpleNamespace(
    # paths
    DATASET=DATASET,
    ROOT_DIR=ROOT_DIR,
    PREPROCESSED_DIR=PREPROCESSED_DIR,
    OUTPUT_DIR=OUTPUT_DIR,
    TRAIN_TSV=TRAIN_TSV,
    DEV_TSV=DEV_TSV,
    TEST_TSV=TEST_TSV,
    NORM_ADJ_PATH=NORM_ADJ_PATH,
    VOCAB_MAP_PATH=VOCAB_MAP_PATH,
    LABELS_PATH=LABELS_PATH,
    DOCS_PATH=DOCS_PATH,
    TRAIN_Y_PATH=TRAIN_Y_PATH,
    VALID_Y_PATH=VALID_Y_PATH,
    TEST_Y_PATH=TEST_Y_PATH,
    TRAIN_Y_PROB_PATH=TRAIN_Y_PROB_PATH,
    VALID_Y_PROB_PATH=VALID_Y_PROB_PATH,
    TEST_Y_PROB_PATH=TEST_Y_PROB_PATH,
    # knobs
    GLOBAL_SEED=GLOBAL_SEED,
    BERT_NAME=BERT_NAME,
    TRANSFORMERS_OFFLINE=TRANSFORMERS_OFFLINE,
    HUGGING_LOCAL_MODEL_FILES_PATH=HUGGING_LOCAL_MODEL_FILES_PATH,
    AUX_MLM=AUX_MLM,
    MLM_PROB=MLM_PROB,
    MLM_LAMBDA=MLM_LAMBDA,
    A3_ENABLE=A3_ENABLE,
    A3_TOPK_ADDR=A3_TOPK_ADDR,
    
    
    A3_UNUSED_BUDGET=A3_UNUSED_BUDGET,
    A3_DT_BINS=A3_DT_BINS,
    A3_VAL_BINS=A3_VAL_BINS,
)
