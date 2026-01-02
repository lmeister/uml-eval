import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from parser import parse_plantuml
from extractor import extract_triples
from matcher import SemanticMatcher

def run_marginal_gain_analysis(json_path, references_dir):
    # Testing range: High to Low (Strictest to Loosest)
    thresholds = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50]
    
    json_path = Path(json_path)
    ref_dir = Path(references_dir)
    
    print(f"Loading experiment results from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data.get('results', [])

    # 1. Pre-load Reference Triples (Performance Optimization)
    reference_cache = {}
    unique_req_ids = set(r['requirement_id'] for r in results)
    print("Pre-loading reference diagrams...")
    
    for req_id in unique_req_ids:
        ref_file = ref_dir / f"{req_id}.puml"
        if ref_file.exists():
            with open(ref_file, 'r', encoding='utf-8') as f:
                parsed = parse_plantuml(f.read())
                reference_cache[req_id] = extract_triples(parsed)

    # Initialize Matcher
    matcher = SemanticMatcher(threshold=1.0)
    
    # Store raw data for every diagram/threshold combo
    raw_data = []

    print(f"Analyzing {len(results)} diagrams across {len(thresholds)} thresholds...")
    
    for entry in tqdm(results):
        req_id = entry['requirement_id']
        if req_id not in reference_cache: continue

        try:
            gen_triples = extract_triples(parse_plantuml(entry['generated_plantuml']))
            ref_triples = reference_cache[req_id]
        except: continue

        # Test every threshold for this specific diagram
        for t in thresholds:
            matcher.threshold = t
            result = matcher.match(gen_triples, ref_triples)
            
            raw_data.append({
                "threshold": t,
                "recall": result.recall,
                "precision": result.precision,
                "f1": result.f1_score
            })

    # 2. Aggregation and Marginal Calculation
    df = pd.DataFrame(raw_data)
    
    # Group by threshold to get averages across all diagrams
    summary = df.groupby("threshold").agg({
        "recall": "mean",
        "precision": "mean",
        "f1": "mean"
    }).reset_index()
    
    # Sort strictly from 1.0 down to 0.50 for step-by-step analysis
    summary = summary.sort_values("threshold", ascending=False).reset_index(drop=True)
    
    # Calculate Cumulative Recall Gain (vs Baseline 1.0)
    baseline_recall = summary[summary["threshold"] == 1.0]["recall"].values[0]
    summary["cumulative_gain"] = summary["recall"] - baseline_recall
    
    # Calculate Marginal Gain (Current Step - Previous Step)
    # shift(1) gives the value of the 'stricter' threshold above it
    summary["prev_recall"] = summary["recall"].shift(1).fillna(baseline_recall)
    summary["marginal_gain"] = summary["recall"] - summary["prev_recall"]
    
    # Cleanup for display
    output_df = summary[["threshold", "recall", "cumulative_gain", "marginal_gain", "precision"]]
    
    print("\n=== MARGINAL RECALL GAIN ANALYSIS ===")
    print(output_df.to_string(index=False, float_format="%.4f"))
    
    # Save to CSV
    output_path = json_path.stem + "_MARGINAL_ANALYSIS.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nAnalysis saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to experiment JSON")
    parser.add_argument("--refs", default="data/references", help="Path to references directory")
    args = parser.parse_args()
    
    run_marginal_gain_analysis(args.json_file, args.refs)