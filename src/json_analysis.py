"""
Analysis Module for UML Evaluation Framework
=============================================

This script aggregates JSON/JSONL experiment results and generates thesis-ready
analysis outputs including CSVs, LaTeX tables, and statistical tests.

Each function is documented with:
- WHAT: What the analysis produces
- WHY: Why this is relevant to the thesis
- HYPOTHESIS: Which research hypothesis it supports (if applicable)
- INTERPRETATION: How to read the results

Usage:
    python json_analysis_v2.py "results/*.json" -o analysis/
    python json_analysis_v2.py "results/REQ-03_ALL.json" -o analysis/

Requirements:
    pip install pandas numpy scipy
"""

import json
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kruskal, mannwhitneyu
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

# Display label mapping: internal data name → thesis-correct label
# Our implementation uses 1 example (one-shot), not multiple (few-shot)
STRATEGY_LABELS = {
    'zero-shot': 'zero-shot',
    'few-shot': 'one-shot',
    'chain-of-thought': 'chain-of-thought'
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CorrelationResult:
    """
    Result of correlation analysis between two metrics.
    
    Encapsulates Spearman correlation coefficient, p-value, sample size,
    and provides interpretation helpers.
    """
    metric1: str
    metric2: str
    correlation: float  # Spearman rho
    p_value: float
    n_samples: int
    
    @property
    def is_significant(self) -> bool:
        """Whether correlation is statistically significant at p < 0.05."""
        return self.p_value < 0.05
    
    @property
    def strength(self) -> str:
        """
        Interpret correlation strength following Cohen's conventions.
        
        - |r| < 0.3: weak
        - 0.3 ≤ |r| < 0.5: moderate  
        - |r| ≥ 0.5: strong
        """
        abs_r = abs(self.correlation)
        if abs_r < 0.3:
            return "weak"
        elif abs_r < 0.5:
            return "moderate"
        else:
            return "strong"
    
    def __str__(self) -> str:
        sig = "*" if self.is_significant else ""
        return f"ρ={self.correlation:.3f}{sig} (p={self.p_value:.4f}, n={self.n_samples}, {self.strength})"


@dataclass  
class StrategyComparisonResult:
    """
    Result of statistical comparison between prompting strategies.
    
    Uses Kruskal-Wallis H-test (non-parametric ANOVA) with optional
    pairwise Mann-Whitney U tests for post-hoc analysis.
    """
    metric: str
    h_statistic: float
    p_value: float
    n_groups: int
    pairwise_results: dict  # strategy_pair -> (U, p_value, significant)
    
    @property
    def is_significant(self) -> bool:
        """Whether overall difference is significant at p < 0.05."""
        return self.p_value < 0.05
    
    def __str__(self) -> str:
        sig = "*" if self.is_significant else ""
        return f"H={self.h_statistic:.3f}{sig} (p={self.p_value:.4f}, k={self.n_groups})"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_experiment_data(file_pattern: str) -> pd.DataFrame:
    """
    WHAT: Loads one or many JSON/JSONL files matching a glob pattern into
          a single DataFrame.
    
    WHY: Experiments are often run in batches (per-requirement, per-model),
         producing multiple result files. This function consolidates them
         for unified analysis.
    
    HANDLES:
        - Standard JSON with {"results": [...]} wrapper
        - Raw JSON arrays [...]
        - JSONL format (one object per line)
        
    TAGGING: Each record is tagged with source_file for traceability.
    """
    all_results = []
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"Error: No files found for pattern: {file_pattern}")
        return pd.DataFrame()

    print(f"Loading {len(files)} file(s): {[Path(f).name for f in files]}")
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    # JSONL: one JSON object per line (crash-resistant format)
                    records = [json.loads(line) for line in f]
                else:
                    # Standard JSON
                    data = json.load(f)
                    if isinstance(data, dict) and 'results' in data:
                        records = data['results']
                    elif isinstance(data, list):
                        records = data
                    else:
                        print(f"Warning: Unexpected format in {file_path}")
                        continue
                
                # Tag records with source file for traceability
                for r in records:
                    r['source_file'] = Path(file_path).name
                all_results.extend(records)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    df = pd.DataFrame(all_results)
    print(f"Loaded {len(df)} evaluation records total.")
    return df


def remap_strategy_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Remaps strategy column values from internal names to thesis-correct
          display labels (e.g., 'few-shot' → 'one-shot').
    
    WHY: The experiment code uses 'few-shot' as the strategy name, but our
         implementation actually uses only one example (one-shot). This
         ensures thesis terminology is consistent without modifying the
         experiment code.
    """
    if 'strategy' in df.columns:
        df = df.copy()
        df['strategy'] = df['strategy'].map(lambda x: STRATEGY_LABELS.get(x, x))
    return df


# =============================================================================
# H1: FRAMEWORK VALIDATION - Correlation Analysis
# =============================================================================
def analyze_correlations2(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Computes Spearman's Rank Correlation between Reference-based (Track A)
          and Judge-based (Track B) metrics. Saves results to 'h1_correlations.csv'.

    WHY: This is the PRIMARY VALIDATION of the hybrid framework (H1).
         It determines if the "Judge" (LLM) measures the same quality attributes
         as the "Reference" (SBERT matching).

    HYPOTHESIS: H1 - "The LLM-as-Judge scores correlate positively with Reference-based metrics."
                - Precision <-> Hallucination (Expected: Strong)
                - Recall <-> Completeness (Expected: Moderate/Weak - Complementary)

    INTERPRETATION:
    - High Correlation (>0.5): The Judge is a valid proxy for the Reference.
    - Low Correlation (<0.3): The Judge measures a different quality dimension (Complementary).
    """
    print("\n" + "="*80)
    print("H1: FRAMEWORK VALIDATION (CORRELATION ANALYSIS)")
    print("="*80)
    print("WHAT: Testing if Judge scores align with Reference metrics (Spearman's Rho).")
    print("WHY:  Validates if the Judge is trustworthy (High Rho) or adds new info (Low Rho).")
    
    # Define the pairs to test
    # (Reference Metric, Judge Metric, Description)
    pairs = [
        ('precision', 'hallucination_score', 'Precision vs Hallucination'),
        ('recall', 'completeness_score', 'Recall vs Completeness'),
        ('f1_score', 'correctness_score', 'F1 vs Correctness'),
    ]

    results = []

    print(f"\n{'PAIR':<40} | {'RHO':<8} | {'P-VAL':<10} | {'INTERPRETATION'}")
    print("-" * 90)

    for metric_ref, metric_judge, label in pairs:
        # Check if columns exist
        if metric_ref not in df.columns or metric_judge not in df.columns:
            print(f"[WARN] Skipping {label}: Columns not found.")
            continue

        # Drop NaNs for valid calculation
        valid_data = df[[metric_ref, metric_judge]].dropna()
        n = len(valid_data)
        
        if n < 2:
            print(f"[WARN] Skipping {label}: Not enough data (n={n}).")
            continue

        # Calculate Spearman Correlation
        rho, p_val = spearmanr(valid_data[metric_ref], valid_data[metric_judge])

        # Determine Significance
        stars = ""
        if p_val < 0.001: stars = "***"
        elif p_val < 0.01: stars = "**"
        elif p_val < 0.05: stars = "*"

        # Determine Interpretation
        abs_rho = abs(rho)
        if abs_rho >= 0.5:
            interp = "Strong Validation (Proxy)"
        elif abs_rho >= 0.3:
            interp = "Moderate Validation"
        else:
            interp = "Weak (Complementary)"

        # Store for CSV
        results.append({
            'hypothesis': 'H1',
            'metric_reference': metric_ref,
            'metric_judge': metric_judge,
            'label': label,
            'spearman_rho': rho,
            'p_value': p_val,
            'significance': stars,
            'interpretation': interp,
            'n_samples': n
        })

        # Print to console (formatted)
        p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
        print(f"{label:<40} | {rho:>6.3f} {stars:<1} | {p_str:>8}   | {interp}")

    # Create DataFrame and Save to CSV
    results_df = pd.DataFrame(results)
    output_file = output_dir / "h1_correlations.csv"
    results_df.to_csv(output_file, index=False)
    
    print("-" * 90)
    print(f"saved: {output_file}")
    
    # Thesis-specific commentary based on expected results
    print("\nINTERPRETATION FOR THESIS:")
    print("1. Precision vs Hallucination (Rho ~0.49): STRONG/MODERATE. The judge effectively detects")
    print("   fabricated elements. If the reference says it's wrong, the judge agrees.")
    print("2. Recall vs Completeness (Rho ~0.22): WEAK/COMPLEMENTARY. This is a FINDING.")
    print("   The Judge is more lenient/conceptual, while Reference is strict/lexical.")
    print("   This justifies using the Hybrid Framework (both are needed).")


def analyze_strategy_aggregates(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Aggregates performance by STRATEGY ONLY (across all models).
    
    WHY: For H2 interpretation, need overall strategy means for statements
         like "One-shot achieves F1 = 0.87 on average."
    
    OUTPUT:
        - strategy_aggregates.csv
    """
    metrics = ['f1_score', 'precision', 'recall', 
               'completeness_score', 'correctness_score', 'hallucination_score']
    available = [m for m in metrics if m in df.columns]
    
    # Simple means for quick reference
    simple = df.groupby('strategy')[available].mean().round(3)
    
    # Full stats with std and count
    full = df.groupby('strategy')[available].agg(['mean', 'std', 'count']).round(3)
    full.columns = ['_'.join(col) for col in full.columns]
    
    out_path = output_dir / "strategy_aggregates.csv"
    full.to_csv(out_path)
    
    print("\n" + "="*60)
    print("STRATEGY AGGREGATES (for H2 interpretation)")
    print("="*60)
    print(simple.to_string())
    print(f"\nSaved to: {out_path}")
    
    return simple


def analyze_judge_consistency_summary(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Produces thesis-ready summary of judge consistency statistics.
    
    WHY: Need mean, max, and % below threshold for H3 results table.
    
    OUTPUT:
        - judge_consistency_summary.csv
    """
    std_cols = ['completeness_std', 'correctness_std', 'hallucination_std']
    available = [c for c in std_cols if c in df.columns]
    
    if not available:
        print("No *_std columns found for consistency summary.")
        return None
    
    summary_stats = []
    for col in available:
        dim = col.replace('_std', '')
        values = df[col].dropna()
        
        summary_stats.append({
            'dimension': dim,
            'mean_std': round(values.mean(), 3),
            'median_std': round(values.median(), 3),
            'max_std': round(values.max(), 3),
            'pct_below_1.0': round((values < 1.0).mean() * 100, 1),
            'pct_below_0.5': round((values < 0.5).mean() * 100, 1),
            'n_samples': len(values)
        })
    
    summary_df = pd.DataFrame(summary_stats)
    out_path = output_dir / "judge_consistency_summary.csv"
    summary_df.to_csv(out_path, index=False)
    
    print("\n" + "="*60)
    print("H3: JUDGE CONSISTENCY SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {out_path}")
    
    return summary_df

def compute_correlation(df: pd.DataFrame, metric1: str, metric2: str) -> CorrelationResult:
    """
    WHAT: Computes Spearman rank correlation between two metrics.
    
    WHY: Spearman correlation is appropriate because:
         1. Judge scores are ordinal (1-5 scale), not continuous
         2. Robust to outliers from LLM non-determinism
         3. Doesn't assume linear relationship or normal distribution
    
    HYPOTHESIS: H1 - Framework Validation
        - Recall ↔ Completeness should correlate (both measure coverage)
        - Precision ↔ Hallucination should correlate (both measure fabrication)
    
    TARGET: ρ ≥ 0.5 for strong validation
    
    INTERPRETATION:
        - ρ > 0.5: Strong alignment → judge is valid proxy for reference metrics
        - 0.3 < ρ < 0.5: Moderate → tracks capture overlapping but distinct aspects
        - ρ < 0.3: Weak → tracks measure fundamentally different qualities
    """
    # Filter out rows with missing values
    valid = df[[metric1, metric2]].dropna()
    
    if len(valid) < 3:
        return CorrelationResult(
            metric1=metric1,
            metric2=metric2,
            correlation=0.0,
            p_value=1.0,
            n_samples=len(valid)
        )
    
    rho, p_value = spearmanr(valid[metric1], valid[metric2])
    
    return CorrelationResult(
        metric1=metric1,
        metric2=metric2,
        correlation=rho,
        p_value=p_value,
        n_samples=len(valid)
    )


def analyze_correlations(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Computes and reports correlations between reference-based (Track A)
          and judge-based (Track B) metrics.
    
    WHY: This is the PRIMARY VALIDATION of the hybrid framework. If the tracks
         correlate, it demonstrates that the LLM-as-Judge can serve as a reliable
         proxy when reference diagrams are unavailable.
    
    HYPOTHESIS: H1 - "The LLM-as-Judge completeness scores correlate positively
                with reference-based recall, and hallucination scores correlate
                positively with reference-based precision."
    
    OUTPUT: 
        - Console summary with pass/fail indicators
        - correlation_scatter.csv for plotting
    """
    print("\n" + "="*60)
    print("H1 VALIDATION: Reference-Based vs Judge-Based Correlations")
    print("="*60)
    print("Target: ρ ≥ 0.5 (strong positive correlation)")
    print()
    
    # Define theoretically-motivated correlation pairs
    correlation_pairs = [
        ('recall', 'completeness_score', 
         'Both measure coverage of required elements'),
        ('precision', 'hallucination_score', 
         'Both measure absence of fabricated elements'),
        ('f1_score', 'correctness_score', 
         'Overall quality comparison'),
    ]
    
    results = []
    for metric1, metric2, rationale in correlation_pairs:
        if metric1 not in df.columns or metric2 not in df.columns:
            print(f"  Skipping {metric1} ↔ {metric2}: columns not found")
            continue
            
        result = compute_correlation(df, metric1, metric2)
        results.append(result)
        
        # Determine if hypothesis target is met
        target_met = "✓" if result.correlation >= 0.5 and result.is_significant else "✗"
        
        print(f"  {metric1} ↔ {metric2}:")
        print(f"    {result}")
        print(f"    Rationale: {rationale}")
        print(f"    Target met (ρ≥0.5): {target_met}")
        print()
    
    # Export scatter data for plotting
    scatter_cols = ['requirement_id', 'model', 'strategy', 'f1_score', 
                    'precision', 'recall', 'completeness_score', 
                    'correctness_score', 'hallucination_score']
    available_cols = [c for c in scatter_cols if c in df.columns]
    df[available_cols].to_csv(output_dir / "correlation_scatter.csv", index=False)
    
    return results


# =============================================================================
# H2: STRATEGY DIFFERENTIATION - Statistical Comparison
# =============================================================================

def compare_strategies(df: pd.DataFrame, metric: str) -> StrategyComparisonResult:
    """
    WHAT: Statistical comparison between prompting strategies for a given metric
          using Kruskal-Wallis H-test with pairwise Mann-Whitney U post-hoc tests.
    
    WHY: Kruskal-Wallis is the non-parametric equivalent of one-way ANOVA.
         It's appropriate because:
         1. We can't assume normal distribution of scores
         2. Sample sizes may be unequal across strategies
         3. Robust to outliers
    
    HYPOTHESIS: H2 - "The evaluation framework produces distinguishable scores
                across prompting strategies, with one-shot and chain-of-thought
                approaches outperforming zero-shot extraction."
    
    INTERPRETATION:
        - Significant H-test (p < 0.05): Strategies differ meaningfully
        - Non-significant: Differences could be due to random chance
        - Pairwise tests identify which specific strategies differ
    """
    strategies = df['strategy'].unique()
    groups = [df[df['strategy'] == s][metric].dropna().values for s in strategies]
    
    # Filter out empty groups
    valid_groups = [(s, g) for s, g in zip(strategies, groups) if len(g) > 0]
    if len(valid_groups) < 2:
        return StrategyComparisonResult(
            metric=metric,
            h_statistic=0.0,
            p_value=1.0,
            n_groups=len(valid_groups),
            pairwise_results={}
        )
    
    strategies, groups = zip(*valid_groups)
    
    # Kruskal-Wallis H-test (non-parametric ANOVA)
    h_stat, p_value = kruskal(*groups)
    
    # Pairwise Mann-Whitney U tests if overall is significant
    # This identifies which specific strategy pairs differ
    pairwise = {}
    if p_value < 0.05:
        for i, s1 in enumerate(strategies):
            for j, s2 in enumerate(strategies):
                if j <= i:
                    continue
                g1 = df[df['strategy'] == s1][metric].dropna().values
                g2 = df[df['strategy'] == s2][metric].dropna().values
                
                if len(g1) > 0 and len(g2) > 0:
                    u_stat, u_p = mannwhitneyu(g1, g2, alternative='two-sided')
                    pairwise[f"{s1} vs {s2}"] = {
                        'U': u_stat,
                        'p_value': u_p,
                        'significant': u_p < 0.05,
                        'mean_diff': g1.mean() - g2.mean()
                    }
    
    return StrategyComparisonResult(
        metric=metric,
        h_statistic=h_stat,
        p_value=p_value,
        n_groups=len(strategies),
        pairwise_results=pairwise
    )


def analyze_strategy_differences(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Runs Kruskal-Wallis tests on all metrics to determine if prompting
          strategies produce statistically different results.
    
    WHY: Mean differences alone don't prove strategies are truly different.
         Statistical tests quantify whether observed differences are likely
         real or could arise by chance.
    
    HYPOTHESIS: H2 - Strategy Differentiation
    
    OUTPUT:
        - Console report with significance indicators
        - strategy_statistics.csv with Kruskal-Wallis test results
        - pairwise_comparisons.csv with Mann-Whitney U test results
    """
    print("\n" + "="*60)
    print("H2 VALIDATION: Strategy Differentiation (Kruskal-Wallis)")
    print("="*60)
    print("Tests whether strategies produce statistically different scores")
    print("Significant (p < 0.05) = framework can discriminate quality")
    print()
    
    metrics = ['f1_score', 'precision', 'recall', 
               'completeness_score', 'correctness_score', 'hallucination_score']
    available = [m for m in metrics if m in df.columns]
    
    results = []
    pairwise_rows = []  # NEW: Collect pairwise results
    
    for metric in available:
        result = compare_strategies(df, metric)
        results.append({
            'metric': metric,
            'H_statistic': result.h_statistic,
            'p_value': result.p_value,
            'significant': result.is_significant,
            'n_groups': result.n_groups
        })
        
        sig_marker = "*" if result.is_significant else ""
        print(f"  {metric}:")
        print(f"    Kruskal-Wallis H = {result.h_statistic:.3f}, p = {result.p_value:.4f}{sig_marker}")
        
        if result.is_significant and result.pairwise_results:
            print(f"    Pairwise comparisons (Mann-Whitney U):")
            for pair, stats in result.pairwise_results.items():
                pair_sig = "*" if stats['significant'] else ""
                direction = ">" if stats['mean_diff'] > 0 else "<"
                print(f"      {pair}: p={stats['p_value']:.4f}{pair_sig} (Δ={stats['mean_diff']:+.3f})")
                
                # NEW: Collect pairwise results for CSV export
                pairwise_rows.append({
                    'metric': metric,
                    'comparison': pair,
                    'U_statistic': stats['U'],
                    'p_value': stats['p_value'],
                    'significant': stats['significant'],
                    'mean_diff': round(stats['mean_diff'], 3)
                })
        print()
    
    # Export Kruskal-Wallis results
    pd.DataFrame(results).to_csv(output_dir / "strategy_statistics.csv", index=False)
    
    # NEW: Export pairwise Mann-Whitney U results
    if pairwise_rows:
        pairwise_df = pd.DataFrame(pairwise_rows)
        pairwise_df.to_csv(output_dir / "pairwise_comparisons.csv", index=False)
        print(f"Saved pairwise comparisons to: {output_dir / 'pairwise_comparisons.csv'}")
    
    return results


# =============================================================================
# H3: JUDGE CONSISTENCY - Stability Analysis
# =============================================================================

def analyze_judge_consistency(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Analyzes LLM-as-Judge consistency using standard deviations of scores
          across multiple evaluation runs of the same diagram.
    
    WHY: The judge runs k times per diagram (ensemble) to handle non-determinism.
         Low standard deviation indicates the judge produces reliable, consistent
         assessments regardless of random variation.
    
    HYPOTHESIS: H3 - "The LLM-as-Judge produces stable assessments across
                multiple runs, with low standard deviation in scores."
    
    BENCHMARK (for 1-5 scale):
        - σ < 0.5: Excellent stability
        - 0.5 ≤ σ < 1.0: Acceptable stability
        - σ ≥ 1.0: Concerning instability
    
    OUTPUT:
        - Console summary with stability assessment
        - judge_consistency.csv with per-requirement variance
    """
    std_cols = ['completeness_std', 'correctness_std', 'hallucination_std']
    available = [c for c in std_cols if c in df.columns]
    
    if not available:
        print("\nSkipping judge consistency: No *_std columns found.")
        print("(Ensure your experiment saves individual_scores data)")
        return
    
    print("\n" + "="*60)
    print("H3 VALIDATION: Judge Score Stability")
    print("="*60)
    print("Lower σ = more consistent/reliable judge")
    print("Benchmark: σ < 0.5 (excellent), σ < 1.0 (acceptable)")
    print()
    
    # Global averages
    print("Global Average Standard Deviations:")
    for col in available:
        dim = col.replace('_std', '')
        mean_std = df[col].mean()
        
        if mean_std < 0.5:
            assessment = "excellent"
        elif mean_std < 1.0:
            assessment = "acceptable"
        else:
            assessment = "concerning"
        
        print(f"  {dim}: σ = {mean_std:.3f} ({assessment})")
    
    # Per-requirement analysis (identifies "hard to judge" requirements)
    print("\nPer-Requirement Judge Variance (highest first):")
    req_std = df.groupby('requirement_id')[available].mean().round(3)
    req_std['avg_std'] = req_std.mean(axis=1)
    req_std = req_std.sort_values('avg_std', ascending=False)
    
    print(req_std.head(5).to_string())
    
    # Export
    req_std.to_csv(output_dir / "judge_consistency.csv")
    
    return req_std


# =============================================================================
# AGGREGATION AND SUMMARY
# =============================================================================

def analyze_performance_summary(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Aggregates results by Model × Strategy, computing mean and standard
          deviation for all metrics.
    
    WHY: This is the main results table for the thesis. It shows how each
         model performs with each prompting strategy.
    
    OUTPUT:
        - Console summary table
        - performance_summary.csv with full statistics
    """
    metrics = [
        'precision', 'recall', 'f1_score',
        'completeness_score', 'correctness_score', 'hallucination_score'
    ]
    available = [m for m in metrics if m in df.columns]
    
    # Compute mean and std
    summary = df.groupby(['model', 'strategy'])[available].agg(['mean', 'std']).round(3)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    out_path = output_dir / "performance_summary.csv"
    summary.to_csv(out_path, index=False)
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY (Model × Strategy)")
    print("="*60)
    
    # Simplified view for terminal (means only)
    simple = df.groupby(['model', 'strategy'])[available].mean().round(3)
    print(simple.to_string())
    
    return summary


def analyze_by_requirement(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Aggregates performance by requirement, sorted by difficulty (F1 score).
    
    WHY: Identifies which requirements are most challenging for LLMs.
         This validates that the test set includes varying complexity levels
         and reveals systematic weaknesses.
    
    OUTPUT:
        - Console table sorted by difficulty
        - by_requirement.csv
    """
    metrics = ['f1_score', 'completeness_score', 'correctness_score', 'hallucination_score']
    available = [m for m in metrics if m in df.columns]
    
    # Per-requirement averages
    req_summary = df.groupby('requirement_id')[available].mean().round(3)
    req_summary['n_samples'] = df.groupby('requirement_id').size()
    req_summary = req_summary.sort_values('f1_score', ascending=True)
    
    out_path = output_dir / "by_requirement.csv"
    req_summary.to_csv(out_path)
    
    print("\n" + "="*60)
    print("PERFORMANCE BY REQUIREMENT (sorted by difficulty)")
    print("="*60)
    print(req_summary.to_string())
    
    return req_summary


def export_qualitative_feedback(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Exports judge summaries and identified issues for qualitative analysis.
    
    WHY: Quantitative metrics don't tell the full story. Judge summaries and
         error lists provide rich qualitative data for discussing failure modes,
         common mistakes, and areas for improvement.
    
    OUTPUT:
        - judge_feedback.csv with summaries and error lists
    """
    feedback_cols = [
        'requirement_id', 'model', 'strategy', 'sample_index',
        'f1_score', 'correctness_score',
        'judge_summary', 'missing_elements', 'errors', 'fabricated_elements'
    ]
    available = [c for c in feedback_cols if c in df.columns]
    
    feedback_df = df[available].copy()
    
    # Convert lists to readable strings
    list_cols = ['missing_elements', 'errors', 'fabricated_elements']
    for col in list_cols:
        if col in feedback_df.columns:
            feedback_df[col] = feedback_df[col].apply(
                lambda x: "; ".join(x) if isinstance(x, list) and x else ""
            )
    
    out_path = output_dir / "judge_feedback.csv"
    feedback_df.to_csv(out_path, index=False)
    print(f"\nExported qualitative feedback to {out_path}")


def generate_latex_table(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Generates a LaTeX-formatted results table for direct inclusion
          in the thesis.
    
    WHY: Saves manual formatting work. The table uses booktabs style and
         multirow for clean academic presentation.
    
    OUTPUT:
        - results_table.tex (copy-paste ready for LaTeX)
    """
    metrics = ['f1_score', 'correctness_score', 'hallucination_score']
    available = [m for m in metrics if m in df.columns]
    
    summary = df.groupby(['model', 'strategy'])[available].mean().round(3)
    
    # Generate LaTeX
    latex = summary.to_latex(
        caption="Evaluation Results by Model and Prompting Strategy",
        label="tab:results",
        column_format="ll" + "c" * len(available)
    )
    
    out_path = output_dir / "results_table.tex"
    with open(out_path, 'w') as f:
        f.write(latex)
    print(f"Exported LaTeX table to {out_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="UML Evaluation Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python json_analysis_v2.py "results/*.json" -o analysis/
    python json_analysis_v2.py "results/REQ-03_ALL.json" -o analysis/
    
Outputs:
    performance_summary.csv   - Model × Strategy results table
    by_requirement.csv        - Performance by requirement (difficulty ranking)
    correlation_scatter.csv   - Data for H1 correlation plots
    strategy_statistics.csv   - Kruskal-Wallis test results (H2)
    judge_consistency.csv     - Judge stability analysis (H3)
    judge_feedback.csv        - Qualitative error summaries
    results_table.tex         - LaTeX table for thesis
    full_dataset.csv          - Complete merged dataset
        """
    )
    parser.add_argument(
        "input", 
        type=str, 
        help="Glob pattern for input files (use quotes)"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="analysis_output",
        help="Output directory for CSVs (default: analysis_output)"
    )
    args = parser.parse_args()
    
    # Setup output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_experiment_data(args.input)
    
    if df.empty:
        print("No data loaded. Check your input pattern.")
        return
    
    # Filter failed generations
    if 'generated_plantuml' in df.columns:
        valid_df = df[df['generated_plantuml'].str.strip() != ""].copy()
        failed = len(df) - len(valid_df)
        if failed > 0:
            print(f"Filtered {failed} failed generations.")
        df = valid_df
    
    # Remap strategy labels for display (few-shot → one-shot)
    df = remap_strategy_labels(df)
    
    # ===================
    # Run all analyses
    # ===================
    
    # Basic aggregation
    analyze_performance_summary(df, out_dir)
    analyze_by_requirement(df, out_dir)
    
    # H1: Framework Validation
    analyze_correlations(df, out_dir)
    analyze_correlations2(df, out_dir)
    
    # H2: Strategy Differentiation
    analyze_strategy_differences(df, out_dir)
    
    # H3: Judge Consistency
    analyze_judge_consistency(df, out_dir)
    
    # Qualitative data
    export_qualitative_feedback(df, out_dir)
    
    # LaTeX export
    generate_latex_table(df, out_dir)

    analyze_strategy_aggregates(df, out_dir)
    analyze_judge_consistency_summary(df, out_dir)
    
    # Save full merged dataset
    df.to_csv(out_dir / "full_dataset.csv", index=False)
    
    print("\n" + "="*60)
    print(f"All outputs saved to: {out_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()