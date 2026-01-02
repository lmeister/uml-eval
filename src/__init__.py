"""
UML Evaluation Framework

A hybrid evaluation framework for assessing LLM-generated UML class diagrams
extracted from natural language requirements.

Components:
- parser: PlantUML parsing
- extractor: Triple extraction
- matcher: Semantic matching with SBERT
- judge: LLM-as-Judge evaluation
- pipeline: Experiment orchestration
"""

# Only import modules that don't have heavy dependencies
from .parser import parse_plantuml, PlantUMLParser, ParsedDiagram
from .extractor import extract_triples, TripleSet, Triple

# Lazy imports for modules with heavy dependencies (SBERT, OpenAI)
def __getattr__(name):
    """Lazy import for modules with heavy dependencies."""
    if name == "SemanticMatcher":
        from .matcher import SemanticMatcher
        return SemanticMatcher
    elif name == "MatchingResult":
        from .matcher import MatchingResult
        return MatchingResult
    elif name == "evaluate_diagram":
        from .matcher import evaluate_diagram
        return evaluate_diagram
    elif name == "LLMJudge":
        from .judge import LLMJudge
        return LLMJudge
    elif name == "JudgeResult":
        from .judge import JudgeResult
        return JudgeResult
    elif name == "evaluate_with_judge":
        from .judge import evaluate_with_judge
        return evaluate_with_judge
    elif name == "Pipeline":
        from .pipeline import Pipeline
        return Pipeline
    elif name == "RequirementData":
        from .pipeline import RequirementData
        return RequirementData
    elif name == "EvaluationResult":
        from .pipeline import EvaluationResult
        return EvaluationResult
    elif name == "config":
        from .config import config
        return config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "0.1.0"
__all__ = [
    "parse_plantuml",
    "extract_triples",
    "SemanticMatcher",
    "LLMJudge",
    "Pipeline",
    "config"
]
