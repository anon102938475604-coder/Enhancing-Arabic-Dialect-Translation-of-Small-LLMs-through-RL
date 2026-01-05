import os
import pandas as pd
import numpy as np
import torch, evaluate
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------
# 0) Repro & caches
# --------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_PARALLELISM"] = "1"
os.environ["HF_HOME"]="/path/to/hf_home"
os.environ["TRANSFORMERS_CACHE"]="/path/to/hf_home"

# --------------------
# 1) Config
# --------------------
MODEL_ID   = "Qwen/Qwen2.5-7B"
DIALECTS = {
    "MSA": "Modern Standard Arabic",
    # Maghreb
    "RAB": "Rabat Arabic", "FES": "Fes Arabic",
    "ALG": "Algiers Arabic",
    "TUN": "Tunis Arabic", "SFX": "Sfax Arabic",
    "TRI": "Tripoli Arabic", "BEN": "Benghazi Arabic",
    # Nile Basin
    "CAI": "Cairo Arabic", "ALX": "Alexandria Arabic",
    "ASW": "Aswan Arabic", "KHA": "Khartoum Arabic",
    # Levant (South & North)
    "JER": "Jerusalem Arabic", "AMM": "Amman Arabic", "SAL": "Salt Arabic",
    "BEI": "Beirut Arabic", "DAM": "Damascus Arabic", "ALE": "Aleppo Arabic",
    # Iraq + Gulf
    "MOS": "Mosul Arabic", "BAG": "Baghdad Arabic", "BAS": "Basra Arabic",
    "DOH": "Doha Arabic", "MUS": "Muscat Arabic",
    "RIY": "Riyadh Arabic", "JED": "Jeddah Arabic",
    # Yemen
    "SAN": "Sana'a Arabic",
}
SEED       = 42
USE_BF16   = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

# Toggle COMET if GPU/weights are available (it's heavy)
USE_COMET = True

# --------------------
# 2) Data prep (wide -> tall for both directions)
# --------------------
DATA_FILE  = f"path/to/full/data.tsv"
OUTPUT_DIR = f"path/to/results/dir"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE, sep="\t")
assert "EN" in df.columns, "Expected EN column in TSV"

# Get dialect columns (skip EN and FR)
start_idx = list(df.columns).index("EN") + 2  # Skip EN and FR columns
dialects = df.columns[start_idx:].tolist()
print(f"Found {len(dialects)} dialects: {dialects}")

# Split wide first (one EN row = one split decision) - same as multidialect SFT
train_val_wide, test_wide = train_test_split(df, test_size=0.10, random_state=SEED, shuffle=True)
# We don't need train/val for zero-shot, but we match the procedure

def make_tall_en_to_dialect(df, dialects):
    """Convert wide format to tall format for EN-to-dialect direction."""
    rows = []
    for _, row in df.iterrows():
        en = row.get("EN")
        if pd.isna(en):
            continue
        for d in dialects:
            ar = row.get(d)
            if pd.isna(ar) or not str(ar).strip():
                continue
            prompt = (
                f"Translate the following English text into {DIALECTS[d]}:\n"
                f"{en}\n"
            )
            rows.append({"dialect": d, "ar": str(ar), "en": str(en), "prompt": prompt})
    return pd.DataFrame(rows)

def make_tall_dialect_to_en(df, dialects):
    """Convert wide format to tall format for dialect-to-EN direction."""
    rows = []
    for _, row in df.iterrows():
        en = row.get("EN")
        if pd.isna(en):
            continue
        for d in dialects:
            ar = row.get(d)
            if pd.isna(ar) or not str(ar).strip():
                continue
            prompt = (
                f"Translate the following {DIALECTS[d]} text into English:\n"
                f"{ar}\n"
            )
            rows.append({"dialect": d, "ar": str(ar), "en": str(en), "prompt": prompt})
    return pd.DataFrame(rows)

# Create tall format test sets for both directions
test_df_en_to_dia = make_tall_en_to_dialect(test_wide, dialects)
test_df_dia_to_en = make_tall_dialect_to_en(test_wide, dialects)

print(f"EN-to-Dialect test size = {len(test_df_en_to_dia)}")
print(f"Dialect-to-EN test size = {len(test_df_dia_to_en)}")

# --------------------
# 3) Load base model (NO adapters)
# --------------------
def prepare_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.model_max_length = 512
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if USE_BF16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")

    # Keep gen/pad/eos/bos consistent (avoids mismatch warnings)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.eval()
    return model, tokenizer

model, tokenizer = prepare_model_and_tokenizer(MODEL_ID)
device = model.device

# --------------------
# 4) Evaluation function
# --------------------
def evaluate_direction(test_df, direction_name, source_col, target_col, ref_col):
    """Evaluate zero-shot performance for one direction."""
    print(f"\n==== Evaluating ZERO-SHOT for {direction_name} ====")
    
    test_prompts = test_df["prompt"].tolist()
    test_refs    = test_df[ref_col].astype(str).tolist()
    test_srcs    = test_df[source_col].astype(str).tolist()

    all_preds = []
    batch_size = 8
    gen_kwargs = dict(
        max_new_tokens=64,
        num_beams=1,
        do_sample=False,          # deterministic baseline
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    for i in range(0, len(test_prompts), batch_size):
        batch_prompts = test_prompts[i:i+batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)

        # Slice generated continuation (after prompt length)
        gen_only = out[:, enc["input_ids"].shape[1]:]
        decoded  = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        batch_preds = [p.strip() for p in decoded]
        all_preds.extend(batch_preds)

    # --------------------
    # 5) Metrics
    # --------------------
    bleu = evaluate.load("sacrebleu")
    bleu_res = bleu.compute(predictions=all_preds, references=[[r] for r in test_refs])

    metrics = {"bleu": float(bleu_res["score"])}

    if USE_COMET:
        try:
            comet = evaluate.load("comet", config_name="Unbabel/wmt22-comet-da")
            comet_res = comet.compute(
                predictions=all_preds,  # MT outputs
                references=test_refs,    # gold refs
                sources=test_srcs,        # sources
            )
            metrics["comet"] = float(comet_res.get("mean_score", comet_res.get("mean", 0.0)))
        except Exception as e:
            print(f"[{direction_name}] WARNING: COMET failed: {e}")

    # Per-dialect breakdown
    per_dialect_metrics = {}
    for d in sorted(test_df["dialect"].unique()):
        mask = (test_df["dialect"] == d).to_numpy()
        preds_d = [p for p, m in zip(all_preds, mask) if m]
        refs_d  = [r for r, m in zip(test_refs, mask) if m]
        srcs_d  = [s for s, m in zip(test_srcs, mask) if m]
        
        if len(preds_d) == 0:
            continue
            
        bleu_d = bleu.compute(predictions=preds_d, references=[[r] for r in refs_d])["score"]
        dial_metrics = {"bleu": bleu_d}
        
        if USE_COMET:
            try:
                comet_d = comet.compute(predictions=preds_d, references=refs_d, sources=srcs_d)["mean_score"]
                dial_metrics["comet"] = comet_d
            except Exception:
                dial_metrics["comet"] = None
        
        per_dialect_metrics[d] = dial_metrics

    return all_preds, metrics, per_dialect_metrics

# --------------------
# 6) Evaluate both directions
# --------------------
# EN-to-Dialect
preds_en_to_dia, metrics_en_to_dia, per_dial_en_to_dia = evaluate_direction(
    test_df_en_to_dia, "EN → Dialect", "en", "ar", "ar"
)

# Dialect-to-EN
preds_dia_to_en, metrics_dia_to_en, per_dial_dia_to_en = evaluate_direction(
    test_df_dia_to_en, "Dialect → EN", "ar", "en", "en"
)

# --------------------
# 7) Save outputs
# --------------------
# EN-to-Dialect results
log_path_en_to_dia = os.path.join(OUTPUT_DIR, "EN_to_Dialect_zeroshot_test_eval.txt")
with open(log_path_en_to_dia, "w", encoding="utf-8") as f:
    f.write("===== ZERO-SHOT TEST SET EVALUATION: EN → Dialect =====\n")
    f.write(f"Overall Test sacreBLEU: {metrics_en_to_dia['bleu']:.2f}\n")
    if "comet" in metrics_en_to_dia:
        f.write(f"Overall Test COMET: {metrics_en_to_dia['comet']:.4f}\n")
    f.write("\nPer-dialect breakdown:\n")
    for d, m in sorted(per_dial_en_to_dia.items()):
        line = f"{d}: BLEU={m['bleu']:.2f}"
        if "comet" in m and m["comet"] is not None:
            line += f", COMET={m['comet']:.4f}"
        f.write(line + "\n")

csv_path_en_to_dia = os.path.join(OUTPUT_DIR, "EN_to_Dialect_zeroshot_test_predictions.csv")
pd.DataFrame({
    "dialect": test_df_en_to_dia["dialect"],
    "source_en": test_df_en_to_dia["en"],
    "reference_ar": test_df_en_to_dia["ar"],
    "prediction_ar": preds_en_to_dia
}).to_csv(csv_path_en_to_dia, index=False, encoding="utf-8")

# Dialect-to-EN results
log_path_dia_to_en = os.path.join(OUTPUT_DIR, "Dialect_to_EN_zeroshot_test_eval.txt")
with open(log_path_dia_to_en, "w", encoding="utf-8") as f:
    f.write("===== ZERO-SHOT TEST SET EVALUATION: Dialect → EN =====\n")
    f.write(f"Overall Test sacreBLEU: {metrics_dia_to_en['bleu']:.2f}\n")
    if "comet" in metrics_dia_to_en:
        f.write(f"Overall Test COMET: {metrics_dia_to_en['comet']:.4f}\n")
    f.write("\nPer-dialect breakdown:\n")
    for d, m in sorted(per_dial_dia_to_en.items()):
        line = f"{d}: BLEU={m['bleu']:.2f}"
        if "comet" in m and m["comet"] is not None:
            line += f", COMET={m['comet']:.4f}"
        f.write(line + "\n")

csv_path_dia_to_en = os.path.join(OUTPUT_DIR, "Dialect_to_EN_zeroshot_test_predictions.csv")
pd.DataFrame({
    "dialect": test_df_dia_to_en["dialect"],
    "source_ar": test_df_dia_to_en["ar"],
    "reference_en": test_df_dia_to_en["en"],
    "prediction_en": preds_dia_to_en
}).to_csv(csv_path_dia_to_en, index=False, encoding="utf-8")

# --------------------
# 8) Print summary
# --------------------
print("\n" + "="*60)
print("===== ZERO-SHOT BIDIRECTIONAL SUMMARY =====")
print("="*60)

print("\n--- EN → Dialect ---")
print(f"Overall BLEU: {metrics_en_to_dia['bleu']:.2f}")
if "comet" in metrics_en_to_dia:
    print(f"Overall COMET: {metrics_en_to_dia['comet']:.4f}")
print("\nPer-dialect:")
for d, m in sorted(per_dial_en_to_dia.items()):
    line = f"  {d}: BLEU={m['bleu']:.2f}"
    if "comet" in m and m["comet"] is not None:
        line += f", COMET={m['comet']:.4f}"
    print(line)

print("\n--- Dialect → EN ---")
print(f"Overall BLEU: {metrics_dia_to_en['bleu']:.2f}")
if "comet" in metrics_dia_to_en:
    print(f"Overall COMET: {metrics_dia_to_en['comet']:.4f}")
print("\nPer-dialect:")
for d, m in sorted(per_dial_dia_to_en.items()):
    line = f"  {d}: BLEU={m['bleu']:.2f}"
    if "comet" in m and m["comet"] is not None:
        line += f", COMET={m['comet']:.4f}"
    print(line)

print("\n" + "="*60)
print("Saved evaluation metrics to:")
print(f"  - {log_path_en_to_dia}")
print(f"  - {log_path_dia_to_en}")
print("Saved detailed test predictions to:")
print(f"  - {csv_path_en_to_dia}")
print(f"  - {csv_path_dia_to_en}")

# Print a few qualitative examples
print("\n--- EN → Dialect Examples ---")
for j in range(min(3, len(test_df_en_to_dia))):
    print(f"\nExample {j+1}:")
    print(f"  EN: {test_df_en_to_dia.iloc[j]['en']}")
    print(f"  REF: {test_df_en_to_dia.iloc[j]['ar']}")
    print(f"  PRED: {preds_en_to_dia[j]}")

print("\n--- Dialect → EN Examples ---")
for j in range(min(3, len(test_df_dia_to_en))):
    print(f"\nExample {j+1}:")
    print(f"  AR: {test_df_dia_to_en.iloc[j]['ar']}")
    print(f"  REF: {test_df_dia_to_en.iloc[j]['en']}")
    print(f"  PRED: {preds_dia_to_en[j]}")

