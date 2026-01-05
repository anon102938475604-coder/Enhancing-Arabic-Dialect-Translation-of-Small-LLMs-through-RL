import os
import json
import pandas as pd
import numpy as np
import torch, evaluate
from sklearn.model_selection import train_test_split
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

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

# --------------------
# 2) Data prep (wide -> text + prompt)
# --------------------
DATA_FILE  = f"path/to/full/data.tsv"
OUTPUT_DIR = f"results/qwen2.5-7b-multidialect-EN-to-dialect"


df = pd.read_csv(DATA_FILE, sep="\t")
assert "EN" in df.columns, "Expected EN column in TSV"

start_idx = list(df.columns).index("EN") + 2 #to skip the FR column
dialects = df.columns[start_idx:]
print(dialects)

# --- Split wide first (one EN row = one split decision) ---
train_val_wide, test_wide = train_test_split(df, test_size=0.10, random_state=SEED, shuffle=True)
train_wide, val_wide      = train_test_split(train_val_wide, test_size=0.1111, random_state=SEED, shuffle=True)


def make_tall(df, dialects):
    rows = []
    for _, row in df.iterrows():
        en = row.get("EN")
        if pd.isna(en): 
            continue
        for d in dialects:
            ar = row.get(d)
            if pd.isna(ar) or not str(ar).strip():
                continue
            train_text = (
                f"Translate the following English text into {DIALECTS[d]}:\n"
                f"<TGT:{d}>\n"
                f"{en}\n"
                f"{ar}"
            )
            prompt = (
                f"Translate the following English text into {DIALECTS[d]}:\n"
                f"<TGT:{d}>\n"
                f"{en}\n"
            )
            rows.append({"dialect": d, "ar": str(ar), "en": str(en),
                         "text": train_text, "prompt": prompt})
    return pd.DataFrame(rows)


train_df = make_tall(train_wide, dialects)
val_df   = make_tall(val_wide, dialects)
test_df  = make_tall(test_wide, dialects)


# --------------------
# 3) Tokenizer & Model
# --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.model_max_length = 512
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

    
# Control tokens: one per target dialect + a fixed source token (optional but fine to keep)
TGT_TOKENS = [f"<TGT:{d}>" for d in dialects]
SPECIAL_TOKENS = TGT_TOKENS

num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
print(f"Added {num_added} special tokens.")

dtype = torch.bfloat16 if USE_BF16 else torch.float16

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
model.resize_token_embeddings(len(tokenizer))  # IMPORTANT after adding tokens
model.gradient_checkpointing_enable()
model.config.use_cache = False
# Then sync with model + generation config
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id

if hasattr(model, "generation_config"):
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id


# make sure we keep BOTH prompt and full text in the dataset
train_ds_raw = Dataset.from_pandas(train_df[["text","prompt"]], preserve_index=False)
val_ds_raw   = Dataset.from_pandas(val_df[["text","prompt"]],   preserve_index=False)

EOS = tokenizer.eos_token_id

def to_masked_features(ex):
    # tokenize FULL text (prompt + target), ensure EOS is present
    full = tokenizer(
        ex["text"],
        truncation=True,
        padding=False,
        max_length=512,
        add_special_tokens=True,
    )
    input_ids = full["input_ids"]
    attn_mask = full["attention_mask"]

    # force EOS at end if not present and there's room
    if input_ids[-1] != EOS and len(input_ids) < 512:
        input_ids = input_ids + [EOS]
        attn_mask = attn_mask + [1]

    # tokenize PROMPT the same way so lengths match tokenization behavior
    pr = tokenizer(
        ex["prompt"],
        truncation=True,
        padding=False,
        max_length=512,
        add_special_tokens=True,
    )
    pr_len = len(pr["input_ids"])

    labels = input_ids.copy()
    labels[:pr_len] = [-100] * pr_len  # mask prompt

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": labels,
    }


train_ds = train_ds_raw.map(to_masked_features, remove_columns=["text","prompt"])
val_ds   = val_ds_raw.map(to_masked_features,   remove_columns=["text","prompt"])


print(f"Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_df)}")  

del train_ds_raw, val_ds_raw



# --------------------
# 4) LoRA config
# --------------------
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
peft_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=TARGET_MODULES,
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")


# --------------------
# 5) Metrics (BLEU for validation)
# --------------------
bleu = evaluate.load("sacrebleu")
comet = None  # Set to None if not using COMET during validation
# comet = evaluate.load("comet", config_name="Unbabel/wmt22-comet-da")

# BEFORE creating the Trainer, grab the sources from your val_df
val_sources = val_df["en"].astype(str).tolist()

def make_compute_metrics(sources_list):
    def compute_metrics(eval_preds):
        # Unpack (works for HF/TRL tuples or objects)
        preds = getattr(eval_preds, "predictions", None)
        labels = getattr(eval_preds, "label_ids", None)
        if preds is None:
            preds, labels = eval_preds

        print("preds type/shape:", type(preds), getattr(preds, "shape", None), flush=True)

        # If logits [B, T, V], convert to token IDs
        if isinstance(preds, np.ndarray) and preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)  # -> [B, T]

        # Some paths return [B,1,T]; squeeze
        if isinstance(preds, np.ndarray) and preds.ndim == 3 and preds.shape[1] == 1:
            preds = preds[:, 0, :]

        def _to_id_seq(row):
            if isinstance(row, np.ndarray):
                row = row.tolist()
            if row and isinstance(row[0], (list, np.ndarray)):  # unwrap [[...]]
                row = row[0]
            return [int(x) for x in row]

        preds_seqs  = [_to_id_seq(r) for r in list(preds)]
        labels_seqs = [_to_id_seq(r) for r in list(labels)]

        # Decode RAW (unsliced) just for visibility
        raw_pred_txt = tokenizer.batch_decode(preds_seqs, skip_special_tokens=True)

        # Build continuation-only slices using label mask (-100 = prompt)
        EOS = tokenizer.eos_token_id

        def slice_continuation(pred_ids, label_ids):
            start = next((j for j, lab in enumerate(label_ids) if lab != -100), len(pred_ids))
            cont = pred_ids[start:]
            if EOS is not None and EOS in cont:
                cont = cont[:cont.index(EOS)]
            return cont

        preds_cont_ids, labels_cont_ids = [], []
        for p_ids, l_ids in zip(preds_seqs, labels_seqs):
            preds_cont_ids.append(slice_continuation(p_ids, l_ids))
            lab = [t for t in l_ids if t != -100]
            if EOS is not None and EOS in lab:
                lab = lab[:lab.index(EOS)]
            labels_cont_ids.append(lab)

        preds_txt  = tokenizer.batch_decode(preds_cont_ids, skip_special_tokens=True)
        labels_txt = tokenizer.batch_decode(labels_cont_ids, skip_special_tokens=True)

        # --- DIAGNOSTICS ---
        k = min(3, len(preds_txt))
        for i in range(k):
            print(f"[SAMPLE {i}] RAW pred[:200]: {raw_pred_txt[i][:200]}", flush=True)
            print(f"[SAMPLE {i}] CONT pred[:200]: {preds_txt[i][:200]}", flush=True)
            print(f"[SAMPLE {i}] LABEL[:200]:    {labels_txt[i][:200]}", flush=True)

        # BLEU on continuation-only
        refs_for_bleu = [[y] for y in labels_txt]
        bleu_res = bleu.compute(predictions=preds_txt, references=refs_for_bleu)
        metrics = {"bleu": float(bleu_res["score"])}

        # COMET with sources (from val_df)
        if comet is not None:
            try:
                srcs = sources_list[:len(preds_txt)]  # slice in case eval is smaller
                comet_res = comet.compute(
                    predictions=preds_txt,
                    references=labels_txt,
                    sources=srcs
                )
                metrics["comet"] = float(comet_res.get("mean_score", comet_res.get("mean", 0.0)))
            except Exception as e:
                print("WARNING: COMET compute failed:", e, flush=True)

        return metrics
    return compute_metrics

# Build your metric fn with val sources
compute_metrics = make_compute_metrics(val_sources)


# --------------------
# 6) Training args
# --------------------
args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=2,
    logging_steps=50,
    eval_strategy="no",   # TRL 0.9.x expects this
    save_strategy="epoch",
    save_total_limit=2,
    eval_accumulation_steps=1,

    fp16=not USE_BF16,
    bf16=USE_BF16,

    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
)

args.predict_with_generate = True
args.generation_max_length = 128
args.generation_num_beams = 1
args.train_on_inputs=False
args.remove_unused_columns = False


# --------------------
# 7) Trainer
# --------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    peft_config=peft_cfg,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=default_data_collator,
    packing=False,
    compute_metrics=compute_metrics,
)

train_output = trainer.train()

#path to save
out_path = os.path.join(OUTPUT_DIR, f"multidialect_train_results.txt")

#save as plain text
with open(out_path, "w", encoding="utf-8") as f:
    f.write(str(train_output))
    

# --------------------
# 8) Save adapter + tokenizer + embedding weights
# --------------------
# Update model config to reflect the resized vocab size (important for loading)
base_model = trainer.model.get_base_model() if hasattr(trainer.model, 'get_base_model') else trainer.model
base_model.config.vocab_size = len(tokenizer)

# Save PEFT adapter
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# IMPORTANT: Save the resized embedding layer weights separately
# because PEFT only saves adapter weights, not the resized embeddings
# Get base model embeddings (PEFT wraps the model, but embeddings are in base model)
embedding_weights = base_model.get_input_embeddings().weight.data.cpu()
torch.save(embedding_weights, os.path.join(OUTPUT_DIR, "embedding_weights.pt"))

# Also save the vocab size in a config file for loading later
vocab_info = {
    "vocab_size": len(tokenizer),
    "original_vocab_size": tokenizer.vocab_size,
    "num_added_tokens": num_added
}
with open(os.path.join(OUTPUT_DIR, "vocab_info.json"), "w") as f:
    json.dump(vocab_info, f, indent=2)

print(f"Saved adapter, tokenizer, embedding weights (vocab_size={len(tokenizer)})")

# --------------------
# 9) Final Test Evaluation (BLEU + sample outputs)
# --------------------
model.eval()
device = model.device

# Prepare test prompts and references
test_prompts = test_df["prompt"].tolist()
test_refs    = test_df["ar"].tolist()
test_srcs = test_df["en"].tolist()


torch.cuda.empty_cache()

# Batch generation
batch_size = 8
all_preds = []

for i in range(0, len(test_prompts), batch_size):
    batch_prompts = test_prompts[i:i+batch_size]
    enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=64,
            num_beams=1,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    # Only take the new tokens after the input prompt length
    gen_only = out[:, enc["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
    batch_preds = [p.strip() for p in decoded]
    all_preds.extend(batch_preds)
    

# Compute BLEU on test set
bleu = evaluate.load("sacrebleu")

comet = evaluate.load("comet", config_name="Unbabel/wmt22-comet-da")

bleu_res = bleu.compute(predictions=all_preds, references=[[r] for r in test_refs])
comet_res = comet.compute(
    predictions=all_preds,   # MT outputs (AR)
    references=test_refs,    # gold refs (AR)
    sources=test_srcs,       # sources (EN)
    # batch_size=8,          # uncomment to limit GPU RAM
)

print(f"\n===== TEST SET EVALUATION =====")
print(f"Test sacreBLEU: {bleu_res['score']:.2f}")
print(f"Test COMET: {comet_res['mean_score']:.4f}")

# Save detailed results
out_path = os.path.join(OUTPUT_DIR, f"multidialect_test_predictions.csv")
pd.DataFrame({
    "dialect": test_df["dialect"],
    "source_en": test_df["en"],
    "reference_ar": test_refs,
    "prediction_ar": all_preds
}).to_csv(out_path, index=False, encoding="utf-8")
print("Saved detailed test predictions to:", out_path)

# Print a few examples
for j in range(3):
    print("\n--- Example", j+1, "---")
    print("EN:", test_df.iloc[j]["en"])
    print("REF:", test_refs[j])
    print("PRED:", all_preds[j])
    
# Per-dialect breakdown (reuses all_preds/test_df order)
per_dial = {}
for d in sorted(test_df["dialect"].unique()):
    mask = (test_df["dialect"] == d).to_numpy()
    preds_d = [p for p, m in zip(all_preds, mask) if m]
    refs_d  = [r for r, m in zip(test_refs, mask) if m]
    srcs_d  = [s for s, m in zip(test_srcs, mask) if m]
    
#     print(d)
#     print(preds_d)
#     print(refs_d)
#     print(srcs_d)

    bleu_d = bleu.compute(predictions=preds_d, references=[[r] for r in refs_d])["score"]
    try:
        comet_d = comet.compute(predictions=preds_d, references=refs_d, sources=srcs_d)["mean_score"]
    except Exception:
        comet_d = None

    per_dial[d] = {"bleu": bleu_d, "comet": comet_d}

print("\nPer-dialect:")
for d, m in per_dial.items():
    print(f"{d}: BLEU={m['bleu']:.2f}" + (f", COMET={m['comet']:.4f}" if m['comet'] is not None else ""))
   

# Save metrics to a text file
log_path = os.path.join(OUTPUT_DIR, f"multidialect_test_eval.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("===== TEST SET EVALUATION =====\n")
    f.write(f"Test overall sacreBLEU: {bleu_res['score']:.2f}\n")
    f.write(f"Test overall COMET: {comet_res['mean_score']:.4f}\n\n")
    for d, m in per_dial.items():
        f.write(f"{d}: BLEU={m['bleu']:.2f}" + (f", COMET={m['comet']:.4f}\n" if m['comet'] is not None else ""))

print("Saved evaluation metrics to:", log_path)
