"""
Thesis Visualization Script for UML Evaluation Framework
=========================================================

This script generates publication-quality figures for the master's thesis.
Each function is documented with:
- WHAT: What the plot shows
- WHY: Why this is relevant to the thesis
- HYPOTHESIS: Which research hypothesis it supports (if applicable)

Usage:
    python thesis_plots.py --input analysis_output/ --output figures/

Requirements:
    pip install matplotlib seaborn pandas scipy numpy
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from typing import Optional

# Thesis-appropriate style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# Color palette - colorblind-friendly
STRATEGY_COLORS = {
    'zero-shot': '#0077BB',      # Blue
    'one-shot': '#EE7733',       # Orange
    'chain-of-thought': '#009988' # Teal
}

# Display label mapping: internal data name → thesis-correct label
STRATEGY_LABELS = {
    'zero-shot': 'Zero-Shot',
    'one-shot': 'One-Shot',
    'chain-of-thought': 'Chain-of-Thought'
}

def get_strategy_label(strategy: str) -> str:
    """Map internal strategy name to display label."""
    return STRATEGY_LABELS.get(strategy, strategy.replace('-', ' ').title())

MODEL_COLORS = sns.color_palette("husl", 8)


# =============================================================================
# H1: FRAMEWORK VALIDATION - Correlation Plots
# =============================================================================

def plot_correlation_scatter(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Three scatter plots showing correlation between Track A (reference-based)
          and Track B (judge-based) metrics.
    
    WHY: This is the PRIMARY VALIDATION of the hybrid framework. If the tracks
         correlate, it demonstrates that the LLM-as-Judge can serve as a reliable
         proxy when reference diagrams are unavailable.
    
    HYPOTHESIS: H1 - "The LLM-as-Judge completeness scores correlate positively 
                with reference-based recall, and hallucination scores correlate 
                positively with reference-based precision."
    
    TARGET: Spearman ρ ≥ 0.5 for strong validation
    
    INTERPRETATION:
        - Strong correlation (ρ > 0.5): Tracks measure similar qualities → validates judge
        - Moderate correlation (0.3-0.5): Partial overlap → complementary information
        - Weak correlation (ρ < 0.3): Tracks capture different aspects → both needed
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Define the correlation pairs with their theoretical justification
    correlation_pairs = [
        {
            'x': 'recall', 
            'y': 'completeness_score',
            'title': 'Recall vs Completeness',
            'rationale': 'Both measure "coverage" of required elements'
        },
        {
            'x': 'precision', 
            'y': 'hallucination_score',
            'title': 'Precision vs Hallucination',
            'rationale': 'Both measure absence of fabricated elements'
        },
        {
            'x': 'f1_score', 
            'y': 'correctness_score',
            'title': 'F1 vs Correctness',
            'rationale': 'Overall quality comparison'
        }
    ]
    
    for ax, pair in zip(axes, correlation_pairs):
        x_col, y_col = pair['x'], pair['y']
        
        # Filter valid data
        valid = df.dropna(subset=[x_col, y_col])
        
        if len(valid) < 5:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            continue
        
        # Calculate correlation
        rho, p_val = spearmanr(valid[x_col], valid[y_col])
        
        # Scatter plot with transparency for overlapping points
        ax.scatter(valid[x_col], valid[y_col], alpha=0.5, s=40, c='#4477AA')
        
        # Add regression line for visual trend
        z = np.polyfit(valid[x_col], valid[y_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2, 
                label=f'ρ = {rho:.3f} (p = {p_val:.4f})')
        
        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_col.replace('_', ' ').title())
        ax.set_title(pair['title'])
        ax.legend(loc='lower right')
        
        # Add interpretation annotation
        if rho >= 0.5:
            interpretation = "Strong alignment"
        elif rho >= 0.3:
            interpretation = "Moderate alignment"
        else:
            interpretation = "Weak alignment"
        
        ax.annotate(interpretation, xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=9, style='italic', va='top')
    
    plt.suptitle('H1 Validation: Reference-Based vs Judge-Based Metric Correlations', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "h1_correlation_scatter.pdf"
    plt.savefig(output_path)
    plt.savefig(output_dir / "h1_correlation_scatter.png")
    print(f"Saved: {output_path}")
    plt.close()


def plot_semantic_gap_quadrants(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: A scatter plot of F1 (x-axis) vs Correctness (y-axis) with quadrant
          annotations highlighting cases where the two tracks diverge.
    
    WHY: This visualization demonstrates the COMPLEMENTARITY of the dual-track
         approach - the core theoretical contribution. Cases in off-diagonal
         quadrants prove that neither track alone is sufficient.
    
    HYPOTHESIS: Supports the overall framework rationale rather than a specific H.
    
    QUADRANT INTERPRETATION:
        - Top-Right (High F1, High Correctness): Agreement - clear success
        - Bottom-Left (Low F1, Low Correctness): Agreement - clear failure  
        - TOP-LEFT (Low F1, High Correctness): Judge recognizes valid alternative
          that reference-based metrics penalize (structural equivalence)
        - BOTTOM-RIGHT (High F1, Low Correctness): Reference matches but judge
          identifies semantic errors (false precision)
    
    The off-diagonal cases are the KEY EVIDENCE for why both tracks are needed.
    """
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    valid = df.dropna(subset=['f1_score', 'correctness_score'])
    
    # Define quadrant boundaries (median or fixed threshold)
    f1_mid = 0.7  # Threshold for "good" F1
    corr_mid = 3.5  # Threshold for "good" correctness (midpoint of 1-5 scale)
    
    # Color points by quadrant
    colors = []
    for _, row in valid.iterrows():
        if row['f1_score'] >= f1_mid and row['correctness_score'] >= corr_mid:
            colors.append('#228B22')  # Green - agreement (success)
        elif row['f1_score'] < f1_mid and row['correctness_score'] < corr_mid:
            colors.append('#DC143C')  # Red - agreement (failure)
        elif row['f1_score'] < f1_mid and row['correctness_score'] >= corr_mid:
            colors.append('#9932CC')  # Purple - judge recognizes valid alternative
        else:
            colors.append('#FF8C00')  # Orange - reference matches but semantic errors
    
    ax.scatter(valid['f1_score'], valid['correctness_score'], 
               c=colors, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Draw quadrant lines
    ax.axvline(x=f1_mid, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=corr_mid, color='gray', linestyle='--', alpha=0.5)
    
    # Quadrant labels
    ax.text(0.35, 4.7, 'Valid Alternative\n(Low F1, High Judge)', 
            ha='center', fontsize=9, style='italic', color='#9932CC')
    ax.text(0.85, 4.7, 'Agreement\n(Both High)', 
            ha='center', fontsize=9, style='italic', color='#228B22')
    ax.text(0.35, 1.5, 'Agreement\n(Both Low)', 
            ha='center', fontsize=9, style='italic', color='#DC143C')
    ax.text(0.85, 1.5, 'Semantic Error\n(High F1, Low Judge)', 
            ha='center', fontsize=9, style='italic', color='#FF8C00')
    
    ax.set_xlabel('F1 Score (Reference-Based)')
    ax.set_ylabel('Correctness Score (Judge-Based)')
    ax.set_title('The Semantic Gap: Where Reference and Judge Metrics Diverge', 
                 fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.8, 5.2)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#228B22', label='Agreement (success)'),
        Patch(facecolor='#DC143C', label='Agreement (failure)'),
        Patch(facecolor='#9932CC', label='Valid alternative'),
        Patch(facecolor='#FF8C00', label='Semantic error'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Count quadrants for annotation
    q_counts = {
        'success': sum(1 for c in colors if c == '#228B22'),
        'failure': sum(1 for c in colors if c == '#DC143C'),
        'valid_alt': sum(1 for c in colors if c == '#9932CC'),
        'semantic_err': sum(1 for c in colors if c == '#FF8C00'),
    }
    
    total = len(colors)
    divergent = q_counts['valid_alt'] + q_counts['semantic_err']
    ax.annotate(f"Divergent cases: {divergent}/{total} ({100*divergent/total:.1f}%)",
               xy=(0.02, 0.02), xycoords='axes fraction', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / "semantic_gap_quadrants.pdf"
    plt.savefig(output_path)
    plt.savefig(output_dir / "semantic_gap_quadrants.png")
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# H2: STRATEGY DIFFERENTIATION - Comparison Plots
# =============================================================================

def plot_strategy_comparison(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Grouped bar chart comparing performance metrics across prompting
          strategies (zero-shot, one-shot, chain-of-thought).
    
    WHY: Demonstrates that the evaluation framework can DISCRIMINATE between
         outputs of varying quality. If all strategies scored similarly, the
         framework would lack sensitivity.
    
    HYPOTHESIS: H2 - "The evaluation framework produces distinguishable scores
                across prompting strategies, with one-shot and chain-of-thought
                approaches outperforming zero-shot extraction."
    
    EXPECTED PATTERN: one-shot ≥ CoT > zero-shot
    (One-shot provides concrete example; CoT adds reasoning but may overthink)
    """
    
    metrics = ['f1_score', 'completeness_score', 'correctness_score', 'hallucination_score']
    available = [m for m in metrics if m in df.columns]
    
    # Aggregate by strategy
    strategy_means = df.groupby('strategy')[available].mean()
    strategy_stds = df.groupby('strategy')[available].std()
    
    # Reorder strategies logically (matching remapped labels from analysis script)
    strategy_order = ['zero-shot', 'one-shot', 'chain-of-thought']
    strategy_order = [s for s in strategy_order if s in strategy_means.index]
    strategy_means = strategy_means.reindex(strategy_order)
    strategy_stds = strategy_stds.reindex(strategy_order)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(available))
    width = 0.25
    
    for i, strategy in enumerate(strategy_order):
        values = strategy_means.loc[strategy, available].values
        errors = strategy_stds.loc[strategy, available].values
        
        bars = ax.bar(x + i * width, values, width, 
                      label=get_strategy_label(strategy),
                      color=STRATEGY_COLORS.get(strategy, f'C{i}'),
                      yerr=errors, capsize=3, alpha=0.85)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('H2 Validation: Performance by Prompting Strategy', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace('_', '\n').title() for m in available])
    ax.legend(title='Strategy')
    ax.set_ylim(0, 5.5)
    
    # Add significance markers placeholder
    ax.annotate('Error bars show ±1 standard deviation', 
               xy=(0.98, 0.02), xycoords='axes fraction',
               ha='right', fontsize=9, style='italic')
    
    plt.tight_layout()
    output_path = output_dir / "h2_strategy_comparison.pdf"
    plt.savefig(output_path)
    plt.savefig(output_dir / "h2_strategy_comparison.png")
    print(f"Saved: {output_path}")
    plt.close()


def plot_strategy_boxplots(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Box plots showing score distributions for each prompting strategy,
          revealing variance and outliers beyond just mean comparisons.
    
    WHY: Mean comparisons can hide important distributional differences.
         A strategy with high mean but high variance may be less reliable
         than one with slightly lower mean but consistent performance.
    
    HYPOTHESIS: H2 - Provides deeper insight into strategy differentiation.
    
    LOOK FOR:
        - Median positions (central tendency)
        - Box heights (interquartile range = consistency)
        - Whisker lengths and outliers (extreme cases)
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    metrics = ['f1_score', 'completeness_score', 'correctness_score', 'hallucination_score']
    titles = ['F1 Score', 'Completeness', 'Correctness', 'Hallucination']
    
    strategy_order = ['zero-shot', 'one-shot', 'chain-of-thought']
    strategy_order = [s for s in strategy_order if s in df['strategy'].unique()]
    
    for ax, metric, title in zip(axes, metrics, titles):
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'{metric} not available', ha='center', va='center')
            continue
        
        # Create box plot
        box_data = [df[df['strategy'] == s][metric].dropna() for s in strategy_order]
        bp = ax.boxplot(box_data, labels=[get_strategy_label(s) for s in strategy_order],
                       patch_artist=True)
        
        # Color boxes by strategy
        for patch, strategy in zip(bp['boxes'], strategy_order):
            patch.set_facecolor(STRATEGY_COLORS.get(strategy, 'gray'))
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Score')
        ax.set_title(title)
        
        # Add sample size annotations
        for i, strategy in enumerate(strategy_order):
            n = len(df[df['strategy'] == strategy][metric].dropna())
            ax.annotate(f'n={n}', xy=(i+1, ax.get_ylim()[0]), 
                       ha='center', fontsize=8, style='italic')
    
    plt.suptitle('Score Distributions by Prompting Strategy', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "h2_strategy_boxplots.pdf"
    plt.savefig(output_path)
    plt.savefig(output_dir / "h2_strategy_boxplots.png")
    print(f"Saved: {output_path}")
    plt.close()


def plot_model_strategy_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Heatmap showing F1 scores (or another metric) for each Model × Strategy
          combination, revealing interaction effects.
    
    WHY: Not all models may benefit equally from advanced prompting strategies.
         Some models might perform well with zero-shot while others need one-shot
         examples. This informs practical deployment decisions.
    
    HYPOTHESIS: H2 - Extends strategy analysis with model-specific patterns.
    
    LOOK FOR:
        - Consistent patterns across rows (strategy effect independent of model)
        - Consistent patterns across columns (model effect independent of strategy)
        - Interaction effects (some models benefit more from certain strategies)
    """
    
    # Pivot table: Model × Strategy → Mean F1
    pivot = df.pivot_table(
        values='f1_score', 
        index='model', 
        columns='strategy', 
        aggfunc='mean'
    )
    
    # Reorder columns
    col_order = ['zero-shot', 'one-shot', 'chain-of-thought']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    
    # Rename columns for display
    pivot.columns = [get_strategy_label(c) for c in pivot.columns]
    
    # Clean model names for display
    pivot.index = pivot.index.str.replace('/', '\n', regex=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.7, vmin=0.4, vmax=1.0,
                ax=ax, cbar_kws={'label': 'F1 Score'})
    
    ax.set_xlabel('Prompting Strategy')
    ax.set_ylabel('Model')
    ax.set_title('Model × Strategy Interaction: F1 Scores', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "h2_model_strategy_heatmap.pdf"
    plt.savefig(output_path)
    plt.savefig(output_dir / "h2_model_strategy_heatmap.png")
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# H3: JUDGE CONSISTENCY - Stability Plots
# =============================================================================

def plot_judge_stability_histogram(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Histogram showing the distribution of standard deviations across
          all judge evaluations, separated by evaluation dimension.
    
    WHY: Proves that the LLM-as-Judge produces RELIABLE assessments. High
         variance would undermine the judge track's credibility as a proxy
         for human evaluation.
    
    HYPOTHESIS: H3 - "The LLM-as-Judge produces stable assessments across
                multiple runs, with low standard deviation in scores."
    
    BENCHMARK: For a 1-5 scale:
        - σ < 0.5: Excellent stability
        - σ 0.5-1.0: Acceptable stability  
        - σ > 1.0: Concerning instability
    
    EXPECTED: Completeness and Hallucination should be more stable than
              Correctness (which requires more subjective judgment).
    """
    
    std_cols = ['completeness_std', 'correctness_std', 'hallucination_std']
    available = [c for c in std_cols if c in df.columns]
    
    if not available:
        print("No standard deviation columns found. Skipping stability histogram.")
        return
    
    fig, axes = plt.subplots(1, len(available), figsize=(4*len(available), 4))
    if len(available) == 1:
        axes = [axes]
    
    colors = ['#4477AA', '#EE6677', '#228833']
    
    for ax, col, color in zip(axes, available, colors):
        data = df[col].dropna()
        
        ax.hist(data, bins=20, color=color, alpha=0.7, edgecolor='white')
        
        # Add vertical lines for benchmarks
        ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.8, label='σ=0.5 (excellent)')
        ax.axvline(x=1.0, color='orange', linestyle='--', alpha=0.8, label='σ=1.0 (acceptable)')
        
        # Add mean line
        mean_std = data.mean()
        ax.axvline(x=mean_std, color='red', linestyle='-', linewidth=2, 
                   label=f'Mean σ={mean_std:.3f}')
        
        dimension = col.replace('_std', '').title()
        ax.set_xlabel('Standard Deviation (σ)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{dimension}')
        ax.legend(fontsize=8)
    
    plt.suptitle('H3 Validation: Judge Score Stability Across Runs', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "h3_judge_stability_histogram.pdf"
    plt.savefig(output_path)
    plt.savefig(output_dir / "h3_judge_stability_histogram.png")
    print(f"Saved: {output_path}")
    plt.close()


def plot_judge_stability_by_dimension(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Box plots comparing standard deviation distributions across the
          three evaluation dimensions (Completeness, Correctness, Hallucination).
    
    WHY: Identifies which aspects of evaluation are more/less reliable.
         If Correctness shows higher variance, this suggests it involves
         more subjective interpretation - important for thesis discussion.
    
    HYPOTHESIS: H3 - Dimension-specific reliability analysis.
    
    THEORETICAL EXPECTATION:
        - Completeness: Low variance (counting present elements is objective)
        - Hallucination: Low variance (detecting extras is relatively clear)
        - Correctness: Higher variance (judging "accuracy" requires interpretation)
    """
    
    std_cols = ['completeness_std', 'correctness_std', 'hallucination_std']
    available = [c for c in std_cols if c in df.columns]
    
    if not available:
        print("No standard deviation columns found. Skipping dimension comparison.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Prepare data for box plot
    box_data = [df[col].dropna() for col in available]
    labels = [col.replace('_std', '').title() for col in available]
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    
    colors = ['#4477AA', '#EE6677', '#228833']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add benchmark lines
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.6, label='Excellent (σ<0.5)')
    ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.6, label='Acceptable (σ<1.0)')
    
    ax.set_ylabel('Standard Deviation (σ)')
    ax.set_xlabel('Evaluation Dimension')
    ax.set_title('Judge Stability by Evaluation Dimension', fontweight='bold')
    ax.legend(loc='upper right')
    
    # Add mean annotations
    for i, (col, label) in enumerate(zip(available, labels)):
        mean_val = df[col].mean()
        ax.annotate(f'μ={mean_val:.2f}', xy=(i+1, mean_val), 
                   xytext=(i+1.3, mean_val),
                   fontsize=9, ha='left')
    
    plt.tight_layout()
    output_path = output_dir / "h3_stability_by_dimension.pdf"
    plt.savefig(output_path)
    plt.savefig(output_dir / "h3_stability_by_dimension.png")
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# COMPLEXITY/DIFFICULTY ANALYSIS
# =============================================================================

def plot_requirement_difficulty(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Horizontal bar chart ranking requirements by difficulty (measured
          by average F1 score), with error bars showing variance.
    
    WHY: Validates that the test set includes requirements of varying complexity.
         A good evaluation framework should be tested across easy and hard cases.
         This also identifies which requirements are most challenging for LLMs.
    
    HYPOTHESIS: Not tied to specific hypothesis - supports experimental design.
    
    LOOK FOR:
        - Spread of difficulties (not all clustered together)
        - Requirements with high variance (inconsistent model performance)
        - Potential correlation with requirement complexity (class count, etc.)
    """
    
    # Calculate per-requirement statistics
    req_stats = df.groupby('requirement_id').agg({
        'f1_score': ['mean', 'std', 'count'],
        'correctness_score': 'mean'
    }).round(3)
    req_stats.columns = ['f1_mean', 'f1_std', 'n_samples', 'correctness_mean']
    req_stats = req_stats.sort_values('f1_mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(req_stats) * 0.5)))
    
    y_pos = np.arange(len(req_stats))
    
    # Color bars by difficulty
    colors = plt.cm.RdYlGn(req_stats['f1_mean'].values)
    
    bars = ax.barh(y_pos, req_stats['f1_mean'], 
                   xerr=req_stats['f1_std'], 
                   capsize=3, color=colors, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(req_stats.index)
    ax.set_xlabel('F1 Score (mean ± std)')
    ax.set_ylabel('Requirement')
    ax.set_title('Requirement Difficulty Ranking (Sorted by F1 Score)', fontweight='bold')
    ax.set_xlim(0, 1.1)
    
    # Add sample count annotations
    for i, (idx, row) in enumerate(req_stats.iterrows()):
        ax.annotate(f"n={int(row['n_samples'])}", 
                   xy=(row['f1_mean'] + row['f1_std'] + 0.02, i),
                   va='center', fontsize=8)
    
    # Add vertical reference lines
    ax.axvline(x=0.5, color='red', linestyle=':', alpha=0.5, label='F1=0.5')
    ax.axvline(x=0.8, color='green', linestyle=':', alpha=0.5, label='F1=0.8')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    output_path = output_dir / "requirement_difficulty.pdf"
    plt.savefig(output_path)
    plt.savefig(output_dir / "requirement_difficulty.png")
    print(f"Saved: {output_path}")
    plt.close()


def plot_requirement_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Heatmap showing all metrics for each requirement, revealing patterns
          of strength and weakness across evaluation dimensions.
    
    WHY: Some requirements may be easy to model completely but hard to model
         correctly. This reveals systematic patterns in model failures.
    
    HYPOTHESIS: Not hypothesis-specific - enriches results analysis.
    
    LOOK FOR:
        - Requirements with consistent performance across metrics (easy/hard)
        - Requirements with divergent performance (e.g., high completeness, low correctness)
        - Patterns suggesting specific failure modes
    """
    
    metrics = ['f1_score', 'completeness_score', 'correctness_score', 'hallucination_score']
    available = [m for m in metrics if m in df.columns]
    
    # Pivot: Requirement → Metrics
    req_metrics = df.groupby('requirement_id')[available].mean()
    
    # Sort by overall performance
    req_metrics['overall'] = req_metrics.mean(axis=1)
    req_metrics = req_metrics.sort_values('overall', ascending=False)
    req_metrics = req_metrics.drop(columns='overall')
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(req_metrics) * 0.4)))
    
    sns.heatmap(req_metrics, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0.7, ax=ax, cbar_kws={'label': 'Score'})
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Requirement')
    ax.set_title('Performance Heatmap by Requirement', fontweight='bold')
    
    # Clean column labels
    ax.set_xticklabels([c.replace('_', '\n').title() for c in req_metrics.columns])
    
    plt.tight_layout()
    output_path = output_dir / "requirement_heatmap.pdf"
    plt.savefig(output_path)
    plt.savefig(output_dir / "requirement_heatmap.png")
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def plot_model_comparison(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Radar chart (or grouped bar) comparing models across all metrics.
    
    WHY: While model comparison is not the primary thesis contribution, it
         demonstrates framework applicability and provides actionable insights.
    
    HYPOTHESIS: Not hypothesis-specific - secondary findings.
    
    NOTE: Keep discussion focused on framework validation rather than making
          strong claims about which model is "best" (not the thesis goal).
    """
    
    metrics = ['f1_score', 'completeness_score', 'correctness_score', 'hallucination_score']
    available = [m for m in metrics if m in df.columns]
    
    # Aggregate by model
    model_means = df.groupby('model')[available].mean()
    
    # Clean model names
    model_means.index = model_means.index.str.split('/').str[-1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_means))
    width = 0.2
    
    for i, metric in enumerate(available):
        bars = ax.bar(x + i * width, model_means[metric], width, 
                      label=metric.replace('_', ' ').title(),
                      alpha=0.85)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.set_xticks(x + width * (len(available)-1) / 2)
    ax.set_xticklabels(model_means.index, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 5.5)
    
    plt.tight_layout()
    output_path = output_dir / "model_comparison.pdf"
    plt.savefig(output_path)
    plt.savefig(output_dir / "model_comparison.png")
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis visualizations for UML evaluation framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python thesis_plots.py --input analysis_output/ --output figures/
    python thesis_plots.py --input analysis_output/full_dataset.csv --output figures/
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing CSVs or path to full_dataset.csv"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="figures",
        help="Output directory for figures (default: figures/)"
    )
    args = parser.parse_args()
    
    # Setup
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if input_path.is_file():
        df = pd.read_csv(input_path)
    else:
        # Look for full_dataset.csv in directory
        dataset_path = input_path / "full_dataset.csv"
        if dataset_path.exists():
            df = pd.read_csv(dataset_path)
        else:
            # Try correlation_scatter.csv as fallback
            scatter_path = input_path / "correlation_scatter.csv"
            if scatter_path.exists():
                df = pd.read_csv(scatter_path)
            else:
                print(f"Error: Could not find data file in {input_path}")
                return
    
    print(f"Loaded {len(df)} records from {input_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate all plots
    print("=" * 60)
    print("GENERATING THESIS FIGURES")
    print("=" * 60)
    
    print("\n--- H1: Framework Validation (Correlations) ---")
    plot_correlation_scatter(df, output_dir)
    plot_semantic_gap_quadrants(df, output_dir)
    
    print("\n--- H2: Strategy Differentiation ---")
    plot_strategy_comparison(df, output_dir)
    plot_strategy_boxplots(df, output_dir)
    if 'model' in df.columns:
        plot_model_strategy_heatmap(df, output_dir)
    
    print("\n--- H3: Judge Consistency ---")
    plot_judge_stability_histogram(df, output_dir)
    plot_judge_stability_by_dimension(df, output_dir)
    
    print("\n--- Complexity Analysis ---")
    plot_requirement_difficulty(df, output_dir)
    plot_requirement_heatmap(df, output_dir)
    
    print("\n--- Model Comparison ---")
    plot_model_comparison(df, output_dir)
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()