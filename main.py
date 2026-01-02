#!/usr/bin/env python3
"""
Main entry point for UML Evaluation Framework.

Usage:
    python main.py --help
    python main.py run --requirements REQ-01 REQ-02 --strategies zero-shot few-shot
    python main.py evaluate --diagram diagram.puml --reference reference.puml
    python main.py test-parser --file diagram.puml
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import config, validate_config
from parser import parse_plantuml, extract_plantuml_from_response
from extractor import extract_triples
from matcher import SemanticMatcher, evaluate_diagram
from judge import LLMJudge
from pipeline import Pipeline, RequirementData, load_requirement_data


def cmd_run(args):
    """Run the full experiment."""
    # Validate configuration
    issues = validate_config()
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        if not args.dry_run:
            sys.exit(1)
    
    # Initialize pipeline
    pipeline = Pipeline()
    
    # Load few-shot examples if using few-shot strategy
    if "few-shot" in args.strategies:
        # Load example requirements (REQ-00)
        try:
            ex1 = load_requirement_data("REQ-00", config.paths.data_dir)
            pipeline.load_few_shot_examples(
                ex1.requirements_text, ex1.reference_plantuml
            )
            print("Loaded few-shot examples: REQ-00")
        except FileNotFoundError as e:
            print(f"Warning: Could not load few-shot examples: {e}")
    
    # Load test requirements
    requirements = []
    for req_id in args.requirements:
        try:
            req_data = load_requirement_data(req_id, config.paths.data_dir)
            requirements.append(req_data)
            print(f"Loaded: {req_id}")
        except FileNotFoundError as e:
            print(f"Warning: Could not load {req_id}: {e}")
    
    if not requirements:
        print("No requirements loaded. Exiting.")
        sys.exit(1)
    
    if args.dry_run:
        print("\nDry run - would evaluate:")
        print(f"  Requirements: {[r.id for r in requirements]}")
        print(f"  Models: {args.models or config.api.extraction_models}")
        print(f"  Strategies: {args.strategies}")
        print(f"  Samples: {args.samples}")
        return
    
    # Run experiment
    results = pipeline.run_experiment(
        requirements=requirements,
        models=args.models,
        strategies=args.strategies,
        samples_per_config=args.samples,
        run_judge=not args.skip_judge,
        verbose=True
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = config.paths.results_dir / f"experiment_{timestamp}.json"
    pipeline.save_results(results, output_path)
    
    # Print summary
    print("\n=== Experiment Summary ===")
    print(f"Total evaluations: {len(results)}")
    
    if results:
        avg_precision = sum(r.precision for r in results) / len(results)
        avg_recall = sum(r.recall for r in results) / len(results)
        avg_f1 = sum(r.f1_score for r in results) / len(results)
        
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"Average F1: {avg_f1:.3f}")
        
        if not args.skip_judge:
            avg_comp = sum(r.completeness_score for r in results) / len(results)
            avg_corr = sum(r.correctness_score for r in results) / len(results)
            avg_hall = sum(r.hallucination_score for r in results) / len(results)
            print(f"Average Completeness: {avg_comp:.2f}")
            print(f"Average Correctness: {avg_corr:.2f}")
            print(f"Average Hallucination: {avg_hall:.2f}")


def cmd_evaluate(args):
    """Evaluate a single diagram against reference."""
    # Load diagrams
    with open(args.diagram, 'r') as f:
        generated = f.read()
    
    with open(args.reference, 'r') as f:
        reference = f.read()
    
    # Parse and extract triples
    gen_parsed = parse_plantuml(generated)
    ref_parsed = parse_plantuml(reference)
    
    gen_triples = extract_triples(gen_parsed)
    ref_triples = extract_triples(ref_parsed)
    
    print(f"Generated: {len(gen_triples)} triples")
    print(f"Reference: {len(ref_triples)} triples")
    
    # Match
    matcher = SemanticMatcher(threshold=args.threshold)
    result = matcher.match(gen_triples, ref_triples)
    
    print(f"\n=== Reference-Based Evaluation ===")
    print(f"Precision: {result.precision:.3f}")
    print(f"Recall: {result.recall:.3f}")
    print(f"F1 Score: {result.f1_score:.3f}")
    
    if args.verbose:
        print(f"\nMatched: {len(result.matched_triples)}")
        for t in result.matched_triples:
            print(f"  {t}")
        
        print(f"\nMissing: {len(result.missing_triples)}")
        for t in result.missing_triples:
            print(f"  {t}")
        
        print(f"\nExcess: {len(result.excess_triples)}")
        for t in result.excess_triples:
            print(f"  {t}")
        
        print(f"\nName Mappings:")
        for gen, ref in result.name_mappings.items():
            score = result.similarity_scores.get(gen, 0)
            if gen != ref:
                print(f"  {gen} -> {ref} (sim={score:.3f})")
    
    # Run judge if requested
    if args.judge and args.requirements:
        with open(args.requirements, 'r') as f:
            requirements = f.read()
        
        issues = validate_config()
        if issues:
            print("\nCannot run judge - configuration issues:")
            for issue in issues:
                print(f"  - {issue}")
            return
        
        judge = LLMJudge(num_runs=args.judge_runs)
        judge_result = judge.evaluate(requirements, generated, verbose=True)
        
        print(f"\n=== LLM-as-Judge Evaluation ===")
        print(f"Completeness: {judge_result.completeness.score:.2f}")
        print(f"Correctness: {judge_result.correctness.score:.2f}")
        print(f"Hallucination: {judge_result.hallucination.score:.2f}")
        print(f"Summary: {judge_result.summary}")


def cmd_test_parser(args):
    """Test the PlantUML parser on a file."""
    with open(args.file, 'r') as f:
        content = f.read()
    
    # Extract PlantUML if embedded in response
    if args.extract:
        content = extract_plantuml_from_response(content)
    
    # Parse
    parsed = parse_plantuml(content)
    
    print(f"=== Parse Results ===")
    print(f"Classes: {len(parsed.classes)}")
    for cls in parsed.classes:
        print(f"  {cls.name}")
        for attr in cls.attributes:
            print(f"    - {attr.name}: {attr.type}")
        for method in cls.methods:
            print(f"    - {method.name}()")
    
    print(f"\nRelationships: {len(parsed.relationships)}")
    for rel in parsed.relationships:
        print(f"  {rel.source} --[{rel.rel_type.value}]--> {rel.target}")
    
    if parsed.parse_errors:
        print(f"\nParse Errors:")
        for err in parsed.parse_errors:
            print(f"  - {err}")
    
    # Extract triples
    if args.triples:
        triples = extract_triples(parsed)
        print(f"\n=== Triples ({len(triples)}) ===")
        for t in triples:
            print(f"  {t}")


def cmd_sensitivity(args):
    """Run sensitivity analysis on threshold parameter."""
    # Load diagrams
    with open(args.diagram, 'r') as f:
        generated = f.read()
    
    with open(args.reference, 'r') as f:
        reference = f.read()
    
    # Parse
    gen_parsed = parse_plantuml(generated)
    ref_parsed = parse_plantuml(reference)
    
    gen_triples = extract_triples(gen_parsed)
    ref_triples = extract_triples(ref_parsed)
    
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    
    print("=== Sensitivity Analysis ===")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 48)
    
    for thresh in thresholds:
        matcher = SemanticMatcher(threshold=thresh)
        result = matcher.match(gen_triples, ref_triples)
        print(f"{thresh:<12.2f} {result.precision:<12.3f} {result.recall:<12.3f} {result.f1_score:<12.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="UML Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Run experiment command
    run_parser = subparsers.add_parser("run", help="Run full experiment")
    run_parser.add_argument(
        "--requirements", "-r",
        nargs="+",
        default=["REQ-01", "REQ-02", "REQ-03", "REQ-04", "REQ-05", "REQ-09", "REQ-10", "REQ-11", "REQ-13"],
        help="Requirement IDs to test (default: REQ-03 through REQ-13)"
    )
    run_parser.add_argument(
        "--models", "-m",
        nargs="+",
        help="Models to test (default: from config)"
    )
    run_parser.add_argument(
        "--strategies", "-s",
        nargs="+",
        default=["zero-shot", "few-shot", "chain-of-thought"],
        help="Prompting strategies"
    )
    run_parser.add_argument(
        "--samples", "-n",
        type=int,
        default=5,
        help="Samples per configuration"
    )
    run_parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-Judge evaluation"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running"
    )
    
    # Evaluate single diagram
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate single diagram")
    eval_parser.add_argument("--diagram", "-d", required=True, help="Generated diagram file")
    eval_parser.add_argument("--reference", "-r", required=True, help="Reference diagram file")
    eval_parser.add_argument("--requirements", "-q", help="Requirements file (for judge)")
    eval_parser.add_argument("--threshold", "-t", type=float, default=0.80, help="Similarity threshold")
    eval_parser.add_argument("--judge", "-j", action="store_true", help="Run LLM-as-Judge")
    eval_parser.add_argument("--judge-runs", type=int, default=3, help="Judge ensemble runs")
    eval_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Test parser
    test_parser = subparsers.add_parser("test-parser", help="Test PlantUML parser")
    test_parser.add_argument("--file", "-f", required=True, help="File to parse")
    test_parser.add_argument("--extract", "-e", action="store_true", help="Extract from LLM response")
    test_parser.add_argument("--triples", "-t", action="store_true", help="Show extracted triples")
    
    # Sensitivity analysis
    sens_parser = subparsers.add_parser("sensitivity", help="Run threshold sensitivity analysis")
    sens_parser.add_argument("--diagram", "-d", required=True, help="Generated diagram")
    sens_parser.add_argument("--reference", "-r", required=True, help="Reference diagram")
    
    args = parser.parse_args()
    
    if args.command == "run":
        cmd_run(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "test-parser":
        cmd_test_parser(args)
    elif args.command == "sensitivity":
        cmd_sensitivity(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
