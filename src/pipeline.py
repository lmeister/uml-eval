"""
Pipeline Orchestrator for UML Evaluation Framework.

This module coordinates the complete evaluation pipeline:
1. Loading requirements and reference diagrams
2. Generating diagrams using LLMs with different prompting strategies
3. Evaluating generated diagrams with both evaluation tracks
4. Aggregating and saving results

Designed to handle the experimental design from Chapter 6:
- 8 test requirements × 3 strategies × 3 models × 5 samples = 360 diagrams
"""

import json
import os
import csv
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from tqdm import tqdm
from openai import OpenAI

from config import config
from parser import parse_plantuml, extract_plantuml_from_response
from extractor import extract_triples, TripleSet
from matcher import SemanticMatcher, MatchingResult
from judge import LLMJudge, JudgeResult


@dataclass
class RequirementData:
    """Data for a single requirement-diagram pair."""
    id: str
    requirements_text: str
    reference_plantuml: str
    reference_triples: Optional[TripleSet] = None
    
    def load_reference_triples(self) -> None:
        """Parse and extract triples from reference diagram."""
        parsed = parse_plantuml(self.reference_plantuml)
        self.reference_triples = extract_triples(parsed)


@dataclass
class GenerationResult:
    """Result of a single diagram generation."""
    requirement_id: str
    model: str
    strategy: str
    sample_index: int
    raw_response: str
    extracted_plantuml: str
    parsed_successfully: bool
    generation_time: float = 0.0


@dataclass  
class EvaluationResult:
    """Complete evaluation result for a single generated diagram."""
    # Generation info
    requirement_id: str
    model: str
    strategy: str
    sample_index: int
    
    # Generated content
    generated_plantuml: str
    
    # Reference-based evaluation
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    matched_count: int = 0
    missing_count: int = 0
    excess_count: int = 0
    
    # LLM-as-Judge evaluation
    completeness_score: float = 0.0
    correctness_score: float = 0.0
    hallucination_score: float = 0.0
    completeness_std: float = 0.0
    correctness_std: float = 0.0
    hallucination_std: float = 0.0
    individual_scores: dict = field(default_factory=dict)

    judge_summary: str = ""
    missing_elements: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    fabricated_elements: list[str] = field(default_factory=list)
    
    # Metadata
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


# Prompting strategy templates
ZERO_SHOT_TEMPLATE = """You are a software engineering expert. Your task is to extract a UML class diagram from the following natural language requirements.

Focus on:
- Identifying classes (key entities/concepts)
- Identifying attributes (properties of classes)
- Identifying methods (behaviors/operations)
- Identifying relationships (associations, inheritance, composition, aggregation)

Output format: Use PlantUML syntax with these constraints:
- Class declarations: class ClassName {{ attributeName : Type; methodName() }}
- Association: ClassA "mult" --> "mult" ClassB
- Inheritance: Child --|> Parent
- Composition: Part *-- Whole
- Aggregation: Part o-- Whole
- NO visibility modifiers, packages, notes, interfaces, or stereotypes

Requirements:
---
{requirements}
---

Provide only the PlantUML code block, no explanations.
"""

FEW_SHOT_TEMPLATE = """You are a software engineering expert. Your task is to extract a UML class diagram from natural language requirements.

Focus on identifying classes, attributes, methods, and relationships.

Output format: PlantUML with these constraints only:
- Class declarations with attributes (name : Type) and methods (name())
- Relationships: --> (association), --|> (inheritance), *-- (composition), o-- (aggregation)
- NO visibility modifiers, packages, notes, interfaces, or stereotypes

Here are examples:

Example 1:
Requirements:
---
{example1_requirements}
---

UML Class Diagram:
{example1_plantuml}

---
Now extract a UML class diagram from the following requirements:
---
{requirements}
---

Provide only the PlantUML code block, no explanations.
"""

COT_TEMPLATE = """You are a software engineering expert. Your task is to extract a UML class diagram from natural language requirements by following these steps explicitly.

Step 1 - Entity Identification:
List all nouns and noun phrases that represent potential classes (concepts, actors, objects).

Step 2 - Entity Classification:
For each candidate, determine if it should be:
- A class (has own properties/behaviors)
- An attribute of another class (simple property)
Explain your reasoning briefly.

Step 3 - Relationship Identification:
Identify how the classes relate to each other. Look for:
- Ownership or containment
- Specialization (is-a relationships)
- Associations (uses, references)

Step 4 - Relationship Typing:
For each relationship, determine the appropriate UML type:
- Association (-->) for general connections
- Inheritance (--|>) for specialization
- Composition (*--) for strong ownership with lifecycle dependency
- Aggregation (o--) for weaker containment
Also note multiplicities if specified.

Step 5 - Diagram Synthesis:
Generate the final PlantUML diagram using this format:
- Class declarations: class ClassName {{ attributeName : Type; methodName() }}
- NO visibility modifiers, packages, notes, interfaces, or stereotypes

Requirements:
---
{requirements}
---

Complete each step, then provide the final PlantUML code block.
"""


class Pipeline:
    """
    Main pipeline orchestrator for UML evaluation experiments.
    """
    
    def __init__(self):
        """Initialize the pipeline with configured components."""
        self.client = OpenAI(
            base_url=config.api.base_url,
            api_key=config.api.api_key
        )
        self.matcher = SemanticMatcher()
        self.judge = LLMJudge()
        
        # Few-shot examples (will be loaded from data)
        self.example1_requirements = ""
        self.example1_plantuml = ""
    
    def load_few_shot_examples(
        self,
        example1_req: str,
        example1_puml: str
    ) -> None:
        """Load few-shot examples."""
        self.example1_requirements = example1_req
        self.example1_plantuml = example1_puml
    
    def _build_prompt(self, strategy: str, requirements: str) -> str:
        """Build prompt for given strategy."""
        if strategy == "zero-shot":
            return ZERO_SHOT_TEMPLATE.format(requirements=requirements)
        
        elif strategy == "few-shot":
            return FEW_SHOT_TEMPLATE.format(
                example1_requirements=self.example1_requirements,
                example1_plantuml=self.example1_plantuml,
                requirements=requirements
            )
        
        elif strategy == "chain-of-thought":
            return COT_TEMPLATE.format(requirements=requirements)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def generate_diagram(
        self,
        requirements: str,
        model: str,
        strategy: str
    ) -> str:
        """
        Generate a UML diagram using specified model and strategy.
        
        Args:
            requirements: Natural language requirements
            model: Model identifier (e.g., "openai/gpt-4o")
            strategy: Prompting strategy name
            
        Returns:
            Generated PlantUML text
        """
        prompt = self._build_prompt(strategy, requirements)
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=config.api.extraction_temperature,
            max_tokens=config.api.max_tokens,
            timeout=90.0 # Um Timeouts zu fixen 
        )
        
        raw_output = response.choices[0].message.content
        return extract_plantuml_from_response(raw_output)
    
    def evaluate_single(
        self,
        requirement_data: RequirementData,
        generated_plantuml: str,
        model: str,
        strategy: str,
        sample_index: int,
        run_judge: bool = True
    ) -> EvaluationResult:
        """
        Evaluate a single generated diagram.
        
        Args:
            requirement_data: Reference data for comparison
            generated_plantuml: Generated diagram text
            model: Model used for generation
            strategy: Strategy used
            sample_index: Sample number within configuration
            run_judge: Whether to run LLM-as-Judge (expensive)
            
        Returns:
            Complete EvaluationResult
        """
        result = EvaluationResult(
            requirement_id=requirement_data.id,
            model=model,
            strategy=strategy,
            sample_index=sample_index,
            generated_plantuml=generated_plantuml,
            timestamp=datetime.now().isoformat()
        )
        
        # Parse and extract triples from generated diagram
        try:
            parsed = parse_plantuml(generated_plantuml)
            generated_triples = extract_triples(parsed)
        except Exception as e:
            print(f"Parse error: {e}")
            return result  # Return with zero scores
        
        # Ensure reference triples are loaded
        if requirement_data.reference_triples is None:
            requirement_data.load_reference_triples()
        
        # Reference-based evaluation
        match_result = self.matcher.match(
            generated_triples,
            requirement_data.reference_triples
        )
        
        result.precision = match_result.precision
        result.recall = match_result.recall
        result.f1_score = match_result.f1_score
        result.matched_count = len(match_result.matched_triples)
        result.missing_count = len(match_result.missing_triples)
        result.excess_count = len(match_result.excess_triples)
        
        # LLM-as-Judge evaluation
        if run_judge:
            judge_result = self.judge.evaluate(
                requirement_data.requirements_text,
                generated_plantuml
            )
            
            result.completeness_score = judge_result.completeness.score
            result.correctness_score = judge_result.correctness.score
            result.hallucination_score = judge_result.hallucination.score
            
            # New fields for variability analysis
            result.completeness_std = judge_result.score_std_devs.get("completeness", 0.0)
            result.correctness_std = judge_result.score_std_devs.get("correctness", 0.0)
            result.hallucination_std = judge_result.score_std_devs.get("hallucination", 0.0)
            
            # Store individual scores from raw responses for deep analysis
            result.individual_scores = {
                "completeness": [r.get("completeness", {}).get("score") for r in judge_result.raw_responses if r],
                "correctness": [r.get("correctness", {}).get("score") for r in judge_result.raw_responses if r],
                "hallucination": [r.get("hallucination", {}).get("score") for r in judge_result.raw_responses if r]
            }

            result.judge_summary = judge_result.summary
            result.missing_elements = judge_result.completeness.issues
            result.errors = judge_result.correctness.issues
            result.fabricated_elements = judge_result.hallucination.issues
        
        return result
    
    def run_experiment(
        self,
        requirements: list[RequirementData],
        models: list[str] = None,
        strategies: list[str] = None,
        samples_per_config: int = None,
        run_judge: bool = True,
        verbose: bool = True
    ) -> list[EvaluationResult]:
        models = models or config.api.extraction_models
        strategies = strategies or ["zero-shot", "few-shot", "chain-of-thought"]
        samples = samples_per_config or config.evaluation.samples_per_config
        
        # --- FIX: Initialize the live backup path here ---
        config.paths.results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_path = config.paths.results_dir / f"live_backup_{timestamp}.jsonl"
        # ------------------------------------------------

        # in case we already tested
        existing_keys = set()
        if jsonl_path.exists():
            with open(jsonl_path, 'r') as f:
                for line in f:
                    res = json.loads(line)
                    # Create a unique key for this configuration
                    key = f"{res['requirement_id']}-{res['strategy']}-{res['model']}-{res['sample_index']}"
                    existing_keys.add(key)

        
        total_configs = len(requirements) * len(strategies) * len(models) * samples
        results = []
        
        if verbose:
            print(f"Running experiment: {total_configs} total evaluations")
            print(f"  Live backup: {jsonl_path}")
        
        progress = tqdm(total=total_configs, disable=not verbose)
        
        for req_data in requirements:
            for strategy in strategies:
                for model in models:
                    for sample_idx in range(samples):
                        try:
                            current_key = f"{req_data.id}-{strategy}-{model}-{sample_idx}"

                            # to help skip after crashes
                            if current_key in existing_keys:
                                progress.update(1)
                                continue # Skip if already done
                            
                            now = datetime.now().strftime("%H:%M:%S")
                            print(f"\nREQUEST [{now}] #{req_data.id} - {strategy} - {model}")
                            
                            plantuml = self.generate_diagram(req_data.requirements_text, model, strategy)
                            
                            # evaluate_single now populates missing_elements, errors, etc.
                            eval_result = self.evaluate_single(
                                req_data, plantuml, model, strategy, sample_idx, run_judge=run_judge
                            )
                            
                            results.append(eval_result)

                            # --- FIXED: Use the locally defined jsonl_path ---
                            with open(jsonl_path, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(eval_result.to_dict()) + "\n")
                            
                            if run_judge:
                                print(f"RESPONSE [{now}] #{req_data.id} - JUDGE (Score: {eval_result.correctness_score})")

                        except Exception as e:
                            print(f"\nError: {req_data.id}/{strategy}/{model}/{sample_idx}: {e}")
                        
                        progress.update(1)
        
        progress.close()
        return results
    
    def save_results(
        self,
        results: list[EvaluationResult],
        output_path: Path
    ) -> None:
        """Save results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "semantic_threshold": config.evaluation.semantic_threshold,
                "sbert_model": config.evaluation.sbert_model,
                "judge_model": config.api.judge_model,
                "samples_per_config": config.evaluation.samples_per_config
            },
            "results": [r.to_dict() for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(results)} results to {output_path}")


def load_requirement_data(req_id: str, data_dir: Path) -> RequirementData:
    """
    Load requirement data from files.
    
    Expected structure:
    - data_dir/requirements/{req_id}.txt
    - data_dir/references/{req_id}.puml
    """
    req_file = data_dir / "requirements" / f"{req_id}.txt"
    ref_file = data_dir / "references" / f"{req_id}.puml"
    
    with open(req_file, 'r') as f:
        requirements_text = f.read()
    
    with open(ref_file, 'r') as f:
        reference_plantuml = f.read()
    
    return RequirementData(
        id=req_id,
        requirements_text=requirements_text,
        reference_plantuml=reference_plantuml
    )


if __name__ == "__main__":
    # Quick test of pipeline components
    print("Testing pipeline components...")
    
    # Test prompt building
    pipeline = Pipeline()
    
    test_req = "A library has books. Each book has a title and author."
    
    print("\nZero-shot prompt (first 200 chars):")
    print(pipeline._build_prompt("zero-shot", test_req)[:200])
    
    print("\nChain-of-thought prompt (first 200 chars):")
    print(pipeline._build_prompt("chain-of-thought", test_req)[:200])
    
    print("\nPipeline components initialized successfully!")
