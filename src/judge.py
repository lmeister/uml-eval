"""
LLM-as-Judge Evaluation for UML Class Diagrams.

This module implements the second evaluation track using a large language
model to assess generated diagrams directly against source requirements.

The judge evaluates three dimensions:
1. Completeness: Are all required entities/relationships present?
2. Correctness: Are modeled elements accurate?
3. Hallucination: Are there elements without basis in requirements?

Uses ensemble approach (multiple runs with majority voting) to mitigate
LLM non-determinism.
"""

import json
from dataclasses import dataclass, field
from typing import Optional
from statistics import mean, stdev
from collections import Counter
from openai import OpenAI

from config import config


# Judge evaluation prompt template
JUDGE_PROMPT_TEMPLATE = """You are an expert software engineer evaluating UML class diagrams extracted from natural language requirements.

## Task
Evaluate the provided UML class diagram against the source requirements. Assess three dimensions:

1. **Completeness** (1-5): Are all entities, attributes, relationships, and methods specified or clearly implied in the requirements present in the diagram?
   - 5: All required elements present
   - 4: Minor omissions (e.g., a single optional attribute)
   - 3: Moderate gaps (e.g., a secondary entity or relationship missing)
   - 2: Significant omissions affecting core entities or multiple relationships
   - 1: Most required elements absent

2. **Correctness** (1-5): Are the modeled elements accurate representations of the requirements?
   - 5: All elements accurately represent requirements (types, multiplicities, relationship kinds)
   - 4: Minor inaccuracies (e.g., imprecise multiplicity)
   - 3: Some errors that don't fundamentally misrepresent the domain
   - 2: Multiple errors or one fundamental misrepresentation
   - 1: Pervasive errors; diagram misrepresents the requirements

3. **Hallucination** (1-5): Higher is better - are there elements that have NO basis in the requirements?
   - 5: No elements without basis in requirements
   - 4: One minor addition that could be considered reasonable inference
   - 3: A few additions not directly derivable from requirements
   - 2: Multiple fabricated elements or one significant invented entity/relationship
   - 1: Extensive fabrication with substantial invented content

## Evaluation Scope
- Focus on functional requirements relevant to class diagram representation
- Ignore non-functional requirements and implementation details not in the source
- The diagram uses a constrained PlantUML format:
  - Classes with attributes (name : Type) and methods (name())
  - Relationships: association (-->), inheritance (--|>), composition (*--), aggregation (o--)
  - No visibility modifiers, interfaces, or abstract class markers
- Do NOT penalize for omitting features outside this constrained format

## Requirements
{requirements}

## Generated UML Class Diagram (PlantUML)
```plantuml
{diagram}
```

## Response Format
Respond with a JSON object only, no other text:
{{
    "completeness": {{
        "score": <1-5>,
        "missing_elements": ["list of missing classes/attributes/relationships"],
        "notes": "brief explanation"
    }},
    "correctness": {{
        "score": <1-5>,
        "errors": ["list of specific errors"],
        "notes": "brief explanation"
    }},
    "hallucination": {{
        "score": <1-5>,
        "fabricated_elements": ["list of elements without basis in requirements"],
        "notes": "brief explanation"
    }},
    "summary": "one sentence overall assessment"
}}
"""


@dataclass
class DimensionResult:
    """Result for a single evaluation dimension."""
    score: float
    issues: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class JudgeResult:
    """Complete result from LLM-as-Judge evaluation."""
    
    completeness: DimensionResult
    correctness: DimensionResult
    hallucination: DimensionResult
    summary: str
    
    # Metadata
    num_runs: int = 1
    score_std_devs: dict = field(default_factory=dict)  # Standard deviations across runs
    raw_responses: list[dict] = field(default_factory=list)
    
    def __str__(self) -> str:
        return (
            f"JudgeResult(\n"
            f"  completeness={self.completeness.score:.2f},\n"
            f"  correctness={self.correctness.score:.2f},\n"
            f"  hallucination={self.hallucination.score:.2f},\n"
            f"  runs={self.num_runs}\n"
            f")"
        )
    
    @property
    def average_score(self) -> float:
        """Average across all three dimensions."""
        return mean([
            self.completeness.score,
            self.correctness.score,
            self.hallucination.score
        ])
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "completeness": {
                "score": self.completeness.score,
                "issues": self.completeness.issues,
                "notes": self.completeness.notes
            },
            "correctness": {
                "score": self.correctness.score,
                "issues": self.correctness.issues,
                "notes": self.correctness.notes
            },
            "hallucination": {
                "score": self.hallucination.score,
                "issues": self.hallucination.issues,
                "notes": self.hallucination.notes
            },
            "summary": self.summary,
            "num_runs": self.num_runs,
            "score_std_devs": self.score_std_devs
        }


class LLMJudge:
    """
    LLM-as-Judge evaluator for UML class diagrams.
    
    Uses ensemble approach with multiple runs and aggregation
    to mitigate non-determinism.
    """
    
    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        num_runs: int = None,
        majority_threshold: int = None
    ):
        """
        Initialize the judge.
        
        Args:
            model: Model to use for judging (default from config)
            temperature: Sampling temperature (default from config)
            num_runs: Number of ensemble runs (default from config)
            majority_threshold: Threshold for majority voting (default from config)
        """
        self.model = model or config.api.judge_model
        self.temperature = temperature if temperature is not None else config.api.judge_temperature
        self.num_runs = num_runs or config.evaluation.judge_runs_per_diagram
        self.majority_threshold = majority_threshold or config.evaluation.majority_threshold
        
        # Initialize API client
        self.client = OpenAI(
            base_url=config.api.base_url,
            api_key=config.api.api_key
        )
    
    def _build_prompt(self, requirements: str, diagram: str) -> str:
        """Build the evaluation prompt."""
        return JUDGE_PROMPT_TEMPLATE.format(
            requirements=requirements,
            diagram=diagram
        )
    
    def _call_judge(self, prompt: str) -> Optional[dict]:
        """
        Make a single judge API call.
        
        Returns:
            Parsed JSON response or None on error
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2048
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
            
       
        except json.JSONDecodeError as e:
            print(f"\n[DEBUG] JSON failed at {e.lineno}:{e.colno}")
            print(f"[DEBUG] Full Content Length: {len(content)}")
            print(f"[DEBUG] Full Raw Content:\n{content}\n") # Print EVERYTHING
            return None
        except Exception as e:
            print(f"API call error: {e}")
            return None
    
    def _aggregate_results(self, responses: list[dict]) -> JudgeResult:
        """
        Aggregate multiple judge responses using averaging and majority voting.
        
        Numerical scores are averaged.
        Categorical outputs (issue lists) use majority voting.
        """
        # Collect scores
        completeness_scores = []
        correctness_scores = []
        hallucination_scores = []
        
        # Collect issues for majority voting
        missing_elements = []
        errors = []
        fabricated_elements = []
        
        # Collect notes and summaries
        completeness_notes = []
        correctness_notes = []
        hallucination_notes = []
        summaries = []
        
        for resp in responses:
            if resp is None:
                continue
            
            # Extract scores
            completeness_scores.append(resp.get("completeness", {}).get("score", 3))
            correctness_scores.append(resp.get("correctness", {}).get("score", 3))
            hallucination_scores.append(resp.get("hallucination", {}).get("score", 3))
            
            # Collect issues
            missing_elements.extend(resp.get("completeness", {}).get("missing_elements", []))
            errors.extend(resp.get("correctness", {}).get("errors", []))
            fabricated_elements.extend(resp.get("hallucination", {}).get("fabricated_elements", []))
            
            # Collect notes
            completeness_notes.append(resp.get("completeness", {}).get("notes", ""))
            correctness_notes.append(resp.get("correctness", {}).get("notes", ""))
            hallucination_notes.append(resp.get("hallucination", {}).get("notes", ""))
            summaries.append(resp.get("summary", ""))
        
        # Compute averages
        avg_completeness = mean(completeness_scores) if completeness_scores else 3.0
        avg_correctness = mean(correctness_scores) if correctness_scores else 3.0
        avg_hallucination = mean(hallucination_scores) if hallucination_scores else 3.0
        
        # Compute standard deviations
        std_devs = {}
        if len(completeness_scores) > 1:
            std_devs["completeness"] = stdev(completeness_scores)
            std_devs["correctness"] = stdev(correctness_scores)
            std_devs["hallucination"] = stdev(hallucination_scores)
        
        # Majority voting for issues
        def majority_vote(items: list, threshold: int) -> list:
            counter = Counter(items)
            return [item for item, count in counter.items() if count >= threshold]
        
        voted_missing = majority_vote(missing_elements, self.majority_threshold)
        voted_errors = majority_vote(errors, self.majority_threshold)
        voted_fabricated = majority_vote(fabricated_elements, self.majority_threshold)
        
        # Select most common note/summary (or first non-empty)
        def select_note(notes: list) -> str:
            non_empty = [n for n in notes if n]
            if not non_empty:
                return ""
            counter = Counter(non_empty)
            return counter.most_common(1)[0][0]
        
        return JudgeResult(
            completeness=DimensionResult(
                score=avg_completeness,
                issues=voted_missing,
                notes=select_note(completeness_notes)
            ),
            correctness=DimensionResult(
                score=avg_correctness,
                issues=voted_errors,
                notes=select_note(correctness_notes)
            ),
            hallucination=DimensionResult(
                score=avg_hallucination,
                issues=voted_fabricated,
                notes=select_note(hallucination_notes)
            ),
            summary=select_note(summaries),
            num_runs=len(responses),
            score_std_devs=std_devs,
            raw_responses=responses
        )
    
    def evaluate(
        self,
        requirements: str,
        diagram: str,
        verbose: bool = False
    ) -> JudgeResult:
        """
        Evaluate a generated diagram against requirements.
        
        Runs multiple evaluations and aggregates results.
        
        Args:
            requirements: Natural language requirements text
            diagram: Generated PlantUML diagram
            verbose: Print progress information
            
        Returns:
            Aggregated JudgeResult
        """
        prompt = self._build_prompt(requirements, diagram)
        
        responses = []
        for i in range(self.num_runs):
            if verbose:
                print(f"  Judge run {i+1}/{self.num_runs}...")
            
            response = self._call_judge(prompt)
            responses.append(response)
        
        # Filter out None responses
        valid_responses = [r for r in responses if r is not None]
        
        if not valid_responses:
            # Return default result on complete failure
            return JudgeResult(
                completeness=DimensionResult(score=0, notes="Evaluation failed"),
                correctness=DimensionResult(score=0, notes="Evaluation failed"),
                hallucination=DimensionResult(score=0, notes="Evaluation failed"),
                summary="Evaluation failed - no valid responses",
                num_runs=self.num_runs,
                raw_responses=responses
            )
        
        return self._aggregate_results(valid_responses)


def create_judge(num_runs: int = None) -> LLMJudge:
    """Create a judge with optional custom number of runs."""
    return LLMJudge(num_runs=num_runs)


def evaluate_with_judge(
    requirements: str,
    diagram: str,
    num_runs: int = None
) -> JudgeResult:
    """
    Convenience function to evaluate a diagram with the LLM judge.
    
    Args:
        requirements: Natural language requirements
        diagram: PlantUML diagram text
        num_runs: Optional number of evaluation runs
        
    Returns:
        JudgeResult with aggregated scores
    """
    judge = create_judge(num_runs=num_runs)
    return judge.evaluate(requirements, diagram)


if __name__ == "__main__":
    # Test (requires API key)
    import os
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY to test the judge")
        exit(1)
    
    test_requirements = """
    A library contains books. Each book has a title, an author, and an ISBN number.
    Books can be borrowed by members. A member has a name and a membership number.
    A member can borrow multiple books at a time, but each book can only be borrowed
    by one member at a time.
    """
    
    test_diagram = """
    @startuml
    class Library {
    }
    
    class Book {
        title : String
        author : String
        isbn : String
    }
    
    class Member {
        name : String
        membershipNumber : int
    }
    
    class Loan {
        borrowDate : Date
    }
    
    Library "1" *-- "0..*" Book : contains
    Member "0..1" --> "0..*" Book : borrows
    @enduml
    """
    
    judge = LLMJudge(num_runs=2)  # Use fewer runs for testing
    result = judge.evaluate(test_requirements, test_diagram, verbose=True)
    
    print(f"\n{result}")
    print(f"\nCompleteness: {result.completeness.score}")
    print(f"  Missing: {result.completeness.issues}")
    print(f"\nCorrectness: {result.correctness.score}")
    print(f"  Errors: {result.correctness.issues}")
    print(f"\nHallucination: {result.hallucination.score}")
    print(f"  Fabricated: {result.hallucination.issues}")
    print(f"\nSummary: {result.summary}")
