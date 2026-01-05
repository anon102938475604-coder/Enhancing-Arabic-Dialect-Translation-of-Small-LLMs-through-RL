#!/usr/bin/env python3
"""
Script to compute per-dialect corpus-level BLEU and COMET scores from inference results.
This matches the training validation approach but uses corpus-level BLEU per dialect.
Also computes COMET scores using Unbabel/wmt22-comet-da model.
"""

import json
import sys
import argparse
from collections import defaultdict
import sacrebleu
from comet import load_from_checkpoint
import os
import glob

# Try to import download_model if available
try:
    from comet import download_model
    HAS_DOWNLOAD_MODEL = True
except ImportError:
    HAS_DOWNLOAD_MODEL = False

def main():
    parser = argparse.ArgumentParser(description="Compute per-dialect BLEU and COMET scores")
    parser.add_argument("--json-file", type=str, required=True, help="JSON file with inference results")
    parser.add_argument("--dataset-name", type=str, default="madar", help="Dataset name prefix (default: madar)")
    parser.add_argument("--output", type=str, help="Output file (default: print to stdout)")
    parser.add_argument("--compute-comet", action="store_true", help="Compute COMET scores in addition to BLEU")
    parser.add_argument("--comet-model", type=str, default="Unbabel/wmt22-comet-da", help="COMET model path (default: Unbabel/wmt22-comet-da)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for COMET computation (default: 64)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs for COMET (default: 1)")
    args = parser.parse_args()
    
    # Load JSON file (supports both JSON and JSONL formats)
    data = []
    with open(args.json_file, 'r', encoding='utf-8') as f:
        # Try to detect if it's JSONL (one JSON object per line) or regular JSON
        first_line = f.readline().strip()
        f.seek(0)  # Reset to beginning
        
        if first_line.startswith('['):
            # Regular JSON array format
            data = json.load(f)
        else:
            # JSONL format (one JSON object per line)
            f.seek(0)  # Reset to beginning
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    
    # Load COMET model only if requested
    comet_model = None
    if args.compute_comet:
        print(f"Loading COMET model: {args.comet_model}")
        
        # Check if it's a HuggingFace model identifier or a local path
        try:
            # Check if it's a local file path
            if os.path.exists(args.comet_model) or args.comet_model.endswith('.ckpt'):
                comet_model = load_from_checkpoint(args.comet_model)
            elif '/' in args.comet_model and HAS_DOWNLOAD_MODEL:
                # Likely a HuggingFace model identifier, try to download it
                print(f"Downloading model from HuggingFace: {args.comet_model}")
                model_path = download_model(args.comet_model)
                print(f"Model downloaded to: {model_path}")
                # The downloaded path might be a directory, check for model.ckpt inside
                if os.path.isdir(model_path):
                    ckpt_path = os.path.join(model_path, "checkpoints", "model.ckpt")
                    if os.path.exists(ckpt_path):
                        comet_model = load_from_checkpoint(ckpt_path)
                    else:
                        # Try looking for .ckpt files in the directory
                        ckpt_files = glob.glob(os.path.join(model_path, "**", "*.ckpt"), recursive=True)
                        if ckpt_files:
                            comet_model = load_from_checkpoint(ckpt_files[0])
                        else:
                            # Try the directory itself
                            comet_model = load_from_checkpoint(model_path)
                else:
                    comet_model = load_from_checkpoint(model_path)
            else:
                # Try loading directly (might work for some HuggingFace paths)
                comet_model = load_from_checkpoint(args.comet_model)
        except Exception as e:
            print(f"Error loading COMET model: {e}")
            print(f"\nThe model path '{args.comet_model}' could not be loaded.")
            if '/' in args.comet_model and not os.path.exists(args.comet_model):
                print("\nFor HuggingFace model identifiers, you have two options:")
                print("1. Download the model first and provide the local path:")
                print("   python -c \"from comet import download_model; print(download_model('Unbabel/wmt22-comet-da'))\"")
                print("2. Or use a local checkpoint path like: ~/wmt22-comet-da/checkpoints/model.ckpt")
            else:
                print("\nPlease provide a valid local checkpoint path (e.g., ~/wmt22-comet-da/checkpoints/model.ckpt)")
            sys.exit(1)
        
        print("COMET model loaded successfully.")
    
    # Group by dialect (data_source)
    # Only collect source texts if COMET is enabled
    if args.compute_comet:
        dialect_data = defaultdict(lambda: {'refs': [], 'hyps': [], 'srcs': []})
    else:
        dialect_data = defaultdict(lambda: {'refs': [], 'hyps': []})
    
    # Determine dataset name from JSON file path or infer from data
    # For madar_en_dialect, data_source format is "madar_en-{dialect}"
    dataset_name = args.dataset_name
    
    for item in data:
        lg = item.get('lg', '')
        if '-' in lg:
            # Construct data_source to match training format: "madar_en-{dialect}"
            # During training, data_source is constructed as: original_data_source + "_" + lg
            # For madar dataset with lg="en-rab", it becomes "madar_en-rab"
            dialect = f"{dataset_name}_{lg}"  # e.g., "madar_en-rab"
            
            ref = item.get('reference_translation', '').strip()
            hyp = item.get('generated_translation', '').strip()
            
            if args.compute_comet:
                src = item.get('source_text', '').strip()
                if ref and hyp and src:
                    dialect_data[dialect]['refs'].append(ref)
                    dialect_data[dialect]['hyps'].append(hyp)
                    dialect_data[dialect]['srcs'].append(src)
            else:
                if ref and hyp:
                    dialect_data[dialect]['refs'].append(ref)
                    dialect_data[dialect]['hyps'].append(hyp)
    
    # Compute corpus-level BLEU and COMET per dialect
    results = {}
    for dialect, texts in dialect_data.items():
        if len(texts['refs']) == 0:
            continue
        
        # Determine tokenization
        # For Arabic dialects, use 13a
        tokenize = "13a"
        
        # Compute corpus-level BLEU
        bleu = sacrebleu.corpus_bleu(
            texts['hyps'],
            [texts['refs']],
            tokenize=tokenize
        )
        
        # Extract BLEU score
        bleu_str = str(bleu)
        bleu_score = float(bleu_str.split("=")[1].split()[0])
        
        # Compute COMET score if requested
        comet_score = None
        if args.compute_comet:
            print(f"Computing COMET for {dialect} ({len(texts['refs'])} examples)...")
            comet_data = [
                {"src": src, "mt": hyp, "ref": ref}
                for src, hyp, ref in zip(texts['srcs'], texts['hyps'], texts['refs'])
            ]
            comet_output = comet_model.predict(comet_data, batch_size=args.batch_size, gpus=args.gpus)
            comet_score = sum(comet_output.scores) / len(comet_output.scores) if comet_output.scores else 0.0
        
        results[dialect] = {
            'bleu': bleu_score,
            'comet': comet_score,
            'count': len(texts['refs'])
        }
    
    # Output results
    output_lines = []
    if args.compute_comet:
        output_lines.append("Per-dialect Corpus-level BLEU and COMET Scores:")
        output_lines.append("=" * 80)
    else:
        output_lines.append("Per-dialect Corpus-level BLEU Scores:")
        output_lines.append("=" * 60)
    
    # Sort by dialect name for consistency
    for dialect in sorted(results.keys()):
        result = results[dialect]
        if args.compute_comet:
            output_lines.append(f"{dialect:20s}: BLEU = {result['bleu']:.2f}, COMET = {result['comet']:.4f} (n={result['count']})")
        else:
            output_lines.append(f"{dialect:20s}: BLEU = {result['bleu']:.2f} (n={result['count']})")
    
    if args.compute_comet:
        output_lines.append("=" * 80)
    else:
        output_lines.append("=" * 60)
    
    # Compute overall corpus-level BLEU and COMET
    all_refs = []
    all_hyps = []
    if args.compute_comet:
        all_srcs = []
        for dialect, texts in dialect_data.items():
            all_refs.extend(texts['refs'])
            all_hyps.extend(texts['hyps'])
            all_srcs.extend(texts['srcs'])
    else:
        for dialect, texts in dialect_data.items():
            all_refs.extend(texts['refs'])
            all_hyps.extend(texts['hyps'])
    
    if all_refs:
        # Overall BLEU
        overall_bleu = sacrebleu.corpus_bleu(
            all_hyps,
            [all_refs],
            tokenize="13a"
        )
        overall_bleu_score = float(str(overall_bleu).split("=")[1].split()[0])
        
        # Overall COMET if requested
        if args.compute_comet:
            print(f"Computing overall COMET ({len(all_refs)} examples)...")
            overall_comet_data = [
                {"src": src, "mt": hyp, "ref": ref}
                for src, hyp, ref in zip(all_srcs, all_hyps, all_refs)
            ]
            overall_comet_output = comet_model.predict(overall_comet_data, batch_size=args.batch_size, gpus=args.gpus)
            overall_comet_score = sum(overall_comet_output.scores) / len(overall_comet_output.scores) if overall_comet_output.scores else 0.0
            output_lines.append(f"{'Overall':20s}: BLEU = {overall_bleu_score:.2f}, COMET = {overall_comet_score:.4f} (n={len(all_refs)})")
        else:
            output_lines.append(f"{'Overall':20s}: BLEU = {overall_bleu_score:.2f} (n={len(all_refs)})")
    
    output_text = "\n".join(output_lines)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Results written to {args.output}")
    else:
        print(output_text)

if __name__ == "__main__":
    main()

