"""
Thesis Visualization Script for UML Evaluation Framework
=========================================================

Generates publication-quality figures for master's thesis validation.

Essential plots (6 core + 1 optional):
1. H1: Correlation scatter plots (validation of hybrid framework)
2. H1: Semantic gap quadrants (complementarity finding - KEY CONTRIBUTION)
3. H2: Strategy comparison (demonstrates framework sensitivity)
4. H2: Strategy boxplots (shows distributions, not just means)
5. H3: Judge stability histogram (proves reliability)
6. Requirement difficulty ranking (validates test set design)
7. [Optional] Model×Strategy heatmap (interaction effects)

Usage:
    python thesis_plots.py --input analysis_output/full_dataset.csv --output figures/

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
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# THESIS STYLE CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# Colorblind-friendly palette
STRATEGY_COLORS = {
    'zero-shot': '#0077BB',           # Blue
    'one-shot': '#EE7733',            # Orange  
    'chain-of-thought': '#009988'     # Teal
}

STRATEGY_LABELS = {
    'zero-shot': 'Zero-Shot',
    'one-shot': 'One-Shot',
    'chain-of-thought': 'Chain-of-Thought'
}

def remap_strategy_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: The experiment code uses 'few-shot' internally, but our implementation
         uses only 1 example (one-shot). This ensures consistent terminology.
    """
    if 'strategy' in df.columns:
        df = df.copy()
        df['strategy'] = df['strategy'].replace({'few-shot': 'one-shot'})
    return df

def get_strategy_label(strategy: str) -> str:
    """Map internal strategy name to thesis display label."""
    return STRATEGY_LABELS.get(strategy, strategy.replace('-', ' ').title())


# =============================================================================
# PLOT 1: H1 VALIDATION - Correlation Scatter Plots
# =============================================================================

def plot_correlation_scatter(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Three scatter plots showing correlations between reference-based
          (Track A) and judge-based (Track B) metrics.
    
    WHY: PRIMARY VALIDATION of hybrid framework (H1). Tests whether the
         LLM-as-Judge can serve as a proxy for reference-based metrics.
    
    HYPOTHESIS: H1 - Target ρ ≥ 0.5 for strong validation
    
    KEY FINDING: Weak Recall-Completeness correlation (ρ=0.22) validates
                 complementarity rather than undermining framework.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    correlation_pairs = [
        ('recall', 'completeness_score', 'Recall vs Completeness'),
        ('precision', 'hallucination_score', 'Precision vs Hallucination'),
        ('f1_score', 'correctness_score', 'F1 vs Correctness'),
    ]
    
    for ax, (x_col, y_col, title) in zip(axes, correlation_pairs):
        valid = df[[x_col, y_col]].dropna()
        
        if len(valid) < 5:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            continue
        
        # Calculate Spearman correlation
        rho, p_val = spearmanr(valid[x_col], valid[y_col])
        
        # Scatter with transparency for overlapping points
        ax.scatter(valid[x_col], valid[y_col], 
                   alpha=0.4, s=30, c='#4477AA', edgecolors='white', linewidth=0.5)
        
        # Regression line for visual trend
        z = np.polyfit(valid[x_col], valid[y_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2)
        
        # Format axis labels
        x_label = x_col.replace('_', ' ').replace(' score', '').title()
        y_label = y_col.replace('_', ' ').replace(' score', '').title()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, fontweight='bold')
        
        # Correlation annotation with interpretation
        if rho >= 0.5:
            interp = "Strong"
            color = '#228B22'
        elif rho >= 0.3:
            interp = "Moderate"
            color = '#FF8C00'
        else:
            interp = "Weak"
            color = '#DC143C'
        
        # P-value formatting
        p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.3f}"
        
        ax.annotate(f'ρ = {rho:.3f}\np {p_str}\n({interp})',
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=10, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color))
    
    plt.suptitle('Framework Validation: Reference vs Judge Metric Correlations (H1)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_dir / "h1_correlation_scatter.pdf")
    plt.savefig(output_dir / "h1_correlation_scatter.png")
    print(f"✓ Saved: h1_correlation_scatter.pdf")
    plt.close()


# =============================================================================
# PLOT 2: H1 - Semantic Gap Quadrants (COMPLEMENTARITY FINDING)
# =============================================================================

def plot_semantic_gap_quadrants(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Scatter plot of F1 (x) vs Correctness (y) with quadrant annotations
          highlighting cases where the two tracks DIVERGE.
    
    WHY: Demonstrates COMPLEMENTARITY - the core theoretical contribution.
         Off-diagonal quadrants prove neither track alone is sufficient.
    
    QUADRANT INTERPRETATION:
        - Top-Right: Agreement (both high) = clear success
        - Bottom-Left: Agreement (both low) = clear failure
        - TOP-LEFT: Low F1 but high correctness = valid alternative structure
        - BOTTOM-RIGHT: High F1 but low correctness = semantic error
    
    The off-diagonal cases are KEY EVIDENCE for hybrid approach.
    """
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    valid = df[['f1_score', 'correctness_score']].dropna()
    
    # Define thresholds (fixed based on typical performance)
    f1_threshold = 0.7   # "Good" F1
    corr_threshold = 3.5  # Midpoint of 1-5 scale
    
    # Assign quadrant colors
    def get_quadrant_color(row):
        if row['f1_score'] >= f1_threshold and row['correctness_score'] >= corr_threshold:
            return '#228B22'  # Green - both agree (success)
        elif row['f1_score'] < f1_threshold and row['correctness_score'] < corr_threshold:
            return '#DC143C'  # Red - both agree (failure)
        elif row['f1_score'] < f1_threshold and row['correctness_score'] >= corr_threshold:
            return '#9932CC'  # Purple - judge sees valid alternative
        else:
            return '#FF8C00'  # Orange - semantic error despite structural match
    
    colors = valid.apply(get_quadrant_color, axis=1)
    
    ax.scatter(valid['f1_score'], valid['correctness_score'],
               c=colors, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Quadrant dividers
    ax.axvline(x=f1_threshold, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=corr_threshold, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Quadrant labels
    ax.text(0.35, 4.7, 'Valid Alternative\n(Low F1, High Judge)',
            ha='center', fontsize=10, style='italic', color='#9932CC', weight='bold')
    ax.text(0.85, 4.7, 'Agreement\n(Both High)',
            ha='center', fontsize=10, style='italic', color='#228B22', weight='bold')
    ax.text(0.35, 1.8, 'Agreement\n(Both Low)',
            ha='center', fontsize=10, style='italic', color='#DC143C', weight='bold')
    ax.text(0.85, 1.8, 'Semantic Error\n(High F1, Low Judge)',
            ha='center', fontsize=10, style='italic', color='#FF8C00', weight='bold')
    
    ax.set_xlabel('F1 Score (Reference-Based)', fontweight='bold')
    ax.set_ylabel('Correctness Score (Judge-Based)', fontweight='bold')
    ax.set_title('The Complementarity Finding: Where Tracks Diverge',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.8, 5.2)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#228B22', label='Agreement (success)'),
        Patch(facecolor='#DC143C', label='Agreement (failure)'),
        Patch(facecolor='#9932CC', label='Valid alternative (complementarity)'),
        Patch(facecolor='#FF8C00', label='Semantic error (complementarity)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    # Count quadrants
    q_counts = colors.value_counts()
    total = len(colors)
    divergent = q_counts.get('#9932CC', 0) + q_counts.get('#FF8C00', 0)
    
    ax.annotate(f"Divergent cases: {divergent}/{total} ({100*divergent/total:.1f}%)\n"
                f"→ Both tracks needed",
               xy=(0.02, 0.02), xycoords='axes fraction', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(output_dir / "h1_semantic_gap_quadrants.pdf")
    plt.savefig(output_dir / "h1_semantic_gap_quadrants.png")
    print(f"✓ Saved: h1_semantic_gap_quadrants.pdf (KEY FINDING)")
    plt.close()


# =============================================================================
# PLOT 3: H2 - Strategy Comparison (Grouped Bar Chart)
# =============================================================================

def plot_strategy_comparison(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Grouped bar chart comparing metrics across prompting strategies.
    
    WHY: Demonstrates framework can DISCRIMINATE quality differences (H2).
         If all strategies scored similarly, framework would lack sensitivity.
    
    HYPOTHESIS: H2 - One-shot should outperform zero-shot
    
    EXPECTED: Error bars show variance; one-shot consistently higher.
    """
    
    metrics = ['f1_score', 'completeness_score', 'correctness_score', 'hallucination_score']
    available = [m for m in metrics if m in df.columns]
    
    strategy_means = df.groupby('strategy')[available].mean()
    strategy_stds = df.groupby('strategy')[available].std()
    
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
        
        ax.bar(x + i * width, values, width,
               label=get_strategy_label(strategy),
               color=STRATEGY_COLORS.get(strategy, f'C{i}'),
               yerr=errors, capsize=3, alpha=0.85, edgecolor='white', linewidth=1)
    
    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Strategy Differentiation: Performance by Prompting Approach (H2)',
                 fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace('_score', '').replace('_', '\n').title() for m in available])
    ax.legend(title='Strategy', framealpha=0.9)
    ax.set_ylim(0, 5.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "h2_strategy_comparison.pdf")
    plt.savefig(output_dir / "h2_strategy_comparison.png")
    print(f"✓ Saved: h2_strategy_comparison.pdf")
    plt.close()


# =============================================================================
# PLOT 4: H2 - Strategy Boxplots (Shows Distributions)
# =============================================================================

def plot_strategy_boxplots(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Box plots showing score distributions for each strategy.
    
    WHY: Mean comparisons hide distributional differences. A strategy with
         high mean but high variance may be less reliable than one with
         slightly lower mean but consistent performance.
    
    LOOK FOR:
        - Median positions (central tendency)
        - Box heights (IQR = consistency)
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
        
        box_data = [df[df['strategy'] == s][metric].dropna() for s in strategy_order]
        bp = ax.boxplot(box_data,
                       labels=[get_strategy_label(s) for s in strategy_order],
                       patch_artist=True, widths=0.6)
        
        # Color boxes
        for patch, strategy in zip(bp['boxes'], strategy_order):
            patch.set_facecolor(STRATEGY_COLORS.get(strategy, 'gray'))
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        # Style whiskers and medians
        for whisker in bp['whiskers']:
            whisker.set(linewidth=1.5, color='black', alpha=0.7)
        for median in bp['medians']:
            median.set(linewidth=2, color='darkred')
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Sample size annotations
        for i, strategy in enumerate(strategy_order):
            n = len(df[df['strategy'] == strategy][metric].dropna())
            y_pos = ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(i+1, y_pos, f'n={n}', ha='center', fontsize=9, style='italic')
    
    plt.suptitle('Score Distributions by Prompting Strategy (H2)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    plt.savefig(output_dir / "h2_strategy_boxplots.pdf")
    plt.savefig(output_dir / "h2_strategy_boxplots.png")
    print(f"✓ Saved: h2_strategy_boxplots.pdf")
    plt.close()


# =============================================================================
# PLOT 5: H3 - Judge Stability Histogram
# =============================================================================

def plot_judge_stability_histogram(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Histogram of standard deviations across judge runs, by dimension.
    
    WHY: Proves LLM-as-Judge produces RELIABLE assessments (H3). High variance
         would undermine judge track credibility.
    
    BENCHMARK (for 1-5 scale):
        - σ < 0.5: Excellent stability
        - σ < 1.0: Acceptable stability
        - σ ≥ 1.0: Concerning instability
    """
    
    std_cols = ['completeness_std', 'correctness_std', 'hallucination_std']
    available = [c for c in std_cols if c in df.columns]
    
    if not available:
        print("⚠ No *_std columns found. Skipping stability histogram.")
        return
    
    fig, axes = plt.subplots(1, len(available), figsize=(4*len(available), 4))
    if len(available) == 1:
        axes = [axes]
    
    colors = ['#4477AA', '#EE6677', '#228833']
    
    for ax, col, color in zip(axes, available, colors):
        data = df[col].dropna()
        
        ax.hist(data, bins=20, color=color, alpha=0.7, edgecolor='white', linewidth=1)
        
        # Benchmark lines
        ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.8, linewidth=2,
                   label='σ=0.5 (excellent)')
        ax.axvline(x=1.0, color='orange', linestyle='--', alpha=0.8, linewidth=2,
                   label='σ=1.0 (acceptable)')
        
        # Mean line
        mean_std = data.mean()
        ax.axvline(x=mean_std, color='red', linestyle='-', linewidth=2.5,
                   label=f'Mean σ={mean_std:.3f}')
        
        dimension = col.replace('_std', '').title()
        ax.set_xlabel('Standard Deviation (σ)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{dimension}', fontweight='bold')
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Judge Score Stability Across 5-Run Ensemble (H3)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_dir / "h3_judge_stability_histogram.pdf")
    plt.savefig(output_dir / "h3_judge_stability_histogram.png")
    print(f"✓ Saved: h3_judge_stability_histogram.pdf")
    plt.close()


# =============================================================================
# PLOT 6: Requirement Difficulty Ranking
# =============================================================================

def plot_requirement_difficulty(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Horizontal bar chart ranking requirements by difficulty (F1 score).
    
    WHY: Validates test set includes varying complexity. A good evaluation
         framework should be tested across easy and hard cases.
    
    LOOK FOR:
        - Spread of difficulties (not all clustered)
        - High-variance requirements (inconsistent performance)
        - Correlation with complexity metrics
    """
    
    req_stats = df.groupby('requirement_id').agg({
        'f1_score': ['mean', 'std', 'count']
    }).round(3)
    req_stats.columns = ['f1_mean', 'f1_std', 'n_samples']
    req_stats = req_stats.sort_values('f1_mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(req_stats) * 0.5)))
    
    y_pos = np.arange(len(req_stats))
    colors = plt.cm.RdYlGn(req_stats['f1_mean'].values)
    
    bars = ax.barh(y_pos, req_stats['f1_mean'],
                   xerr=req_stats['f1_std'],
                   capsize=3, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(req_stats.index)
    ax.set_xlabel('F1 Score (Mean ± Std)', fontweight='bold')
    ax.set_ylabel('Requirement', fontweight='bold')
    ax.set_title('Requirement Difficulty Ranking (Sorted by F1)',
                 fontweight='bold')
    ax.set_xlim(0, 1.1)
    
    # Sample count annotations
    for i, (idx, row) in enumerate(req_stats.iterrows()):
        ax.annotate(f"n={int(row['n_samples'])}",
                   xy=(row['f1_mean'] + row['f1_std'] + 0.02, i),
                   va='center', fontsize=9)
    
    # Reference lines
    ax.axvline(x=0.5, color='red', linestyle=':', alpha=0.5, linewidth=2, label='F1=0.5')
    ax.axvline(x=0.8, color='green', linestyle=':', alpha=0.5, linewidth=2, label='F1=0.8')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "requirement_difficulty.pdf")
    plt.savefig(output_dir / "requirement_difficulty.png")
    print(f"✓ Saved: requirement_difficulty.pdf")
    plt.close()


# =============================================================================
# PLOT 7 (OPTIONAL): Model × Strategy Heatmap
# =============================================================================

def plot_model_strategy_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    WHAT: Heatmap showing F1 for each Model × Strategy combination.
    
    WHY: Reveals interaction effects. Not all models benefit equally from
         advanced prompting. Informs practical deployment decisions.
    
    USE CASE: Include if discussing model-specific patterns in detail.
              Otherwise, omit to keep figure count manageable.
    """
    
    pivot = df.pivot_table(
        values='f1_score',
        index='model',
        columns='strategy',
        aggfunc='mean'
    )
    
    col_order = ['zero-shot', 'one-shot', 'chain-of-thought']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    pivot.columns = [get_strategy_label(c) for c in pivot.columns]
    
    # Clean model names
    pivot.index = pivot.index.str.split('/').str[-1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0.7, vmin=0.4, vmax=1.0,
                ax=ax, cbar_kws={'label': 'F1 Score'},
                linewidths=0.5, linecolor='white')
    
    ax.set_xlabel('Prompting Strategy', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    ax.set_title('Model × Strategy Interaction Effects',
                 fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "h2_model_strategy_heatmap.pdf")
    plt.savefig(output_dir / "h2_model_strategy_heatmap.png")
    print(f"✓ Saved: h2_model_strategy_heatmap.pdf (optional)")
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
    python thesis_plots.py --input analysis_output/full_dataset.csv --output figures/
    python thesis_plots.py -i results.csv -o figs/ --skip-optional
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to full_dataset.csv or analysis directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="figures",
        help="Output directory for figures (default: figures/)"
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional plots (model×strategy heatmap)"
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
        dataset_path = input_path / "full_dataset.csv"
        if dataset_path.exists():
            df = pd.read_csv(dataset_path)
        else:
            print(f"❌ Error: Could not find data file in {input_path}")
            return
    
    print(f"\n{'='*60}")
    print(f"UML EVALUATION FRAMEWORK - THESIS FIGURES")
    print(f"{'='*60}")
    print(f"Loaded {len(df)} records")
    print(f"Output: {output_dir.absolute()}\n")
    
    # Remap strategy labels
    df = remap_strategy_column(df)
    
    # Generate core plots
    print("GENERATING CORE PLOTS (6 essential)")
    print("-" * 60)
    
    print("\n[1/6] H1: Correlation scatter plots...")
    plot_correlation_scatter(df, output_dir)
    
    print("[2/6] H1: Semantic gap quadrants (KEY FINDING)...")
    plot_semantic_gap_quadrants(df, output_dir)
    
    print("[3/6] H2: Strategy comparison...")
    plot_strategy_comparison(df, output_dir)
    
    print("[4/6] H2: Strategy boxplots...")
    plot_strategy_boxplots(df, output_dir)
    
    print("[5/6] H3: Judge stability...")
    plot_judge_stability_histogram(df, output_dir)
    
    print("[6/6] Requirement difficulty...")
    plot_requirement_difficulty(df, output_dir)
    
    # Optional plot
    if not args.skip_optional and 'model' in df.columns:
        print("\nGENERATING OPTIONAL PLOTS")
        print("-" * 60)
        print("[Optional] Model × Strategy heatmap...")
        plot_model_strategy_heatmap(df, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ All figures saved to: {output_dir.absolute()}")
    print(f"{'='*60}\n")
    
    print("FIGURE CHECKLIST FOR THESIS:")
    print("  □ h1_correlation_scatter.pdf (3 subplots)")
    print("  □ h1_semantic_gap_quadrants.pdf (COMPLEMENTARITY)")
    print("  □ h2_strategy_comparison.pdf")
    print("  □ h2_strategy_boxplots.pdf")
    print("  □ h3_judge_stability_histogram.pdf")
    print("  □ requirement_difficulty.pdf")
    if not args.skip_optional:
        print("  □ h2_model_strategy_heatmap.pdf (optional)\n")


if __name__ == "__main__":
    main()