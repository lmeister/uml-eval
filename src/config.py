"""
Configuration settings for the UML Evaluation Framework.

This module centralizes all configurable parameters including:
- API endpoints and model selection
- Semantic matching thresholds
- Evaluation parameters
- File paths
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class APIConfig:
    """Configuration for API access (OpenRouter or direct providers)."""
    
    # OpenRouter provides unified access to multiple LLM providers
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    
    # Extraction models to test
    extraction_models: list = field(default_factory=lambda: [
        "google/gemini-2.5-flash",
        "deepseek/deepseek-v3.2",
        "openai/gpt-4o-mini",
        "z-ai/glm-4-32b",
        "mistralai/devstral-2512:free"
    ])
    
    # Judge model 
    judge_model: str = "x-ai/grok-code-fast-1"
    
    # Generation parameters
    extraction_temperature: float = 0.5  # Balance diversity vs consistency
    judge_temperature: float = 0.3       # Lower for more consistent judgments
    max_tokens: int = 4096


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""
    
    # Semantic matching threshold for SBERT cosine similarity
    # 0.80 is our starting point; sensitivity analysis tests 0.70-0.90
    semantic_threshold: float = 0.80
    
    # SBERT model for embeddings
    sbert_model: str = "all-MiniLM-L6-v2"
    
    # Number of samples per configuration (addresses non-determinism)
    samples_per_config: int = 5
    
    # Number of judge runs per diagram (for ensemble averaging)
    judge_runs_per_diagram: int = 5
    
    # Majority voting threshold for categorical outputs
    majority_threshold: int = 3  # 3 out of 5 runs must agree


@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"
    
    @property
    def requirements_dir(self) -> Path:
        return self.data_dir / "requirements"
    
    @property
    def references_dir(self) -> Path:
        return self.data_dir / "references"
    
    @property
    def results_dir(self) -> Path:
        return self.base_dir / "results"
    
    @property
    def prompts_dir(self) -> Path:
        return self.base_dir / "prompts"


@dataclass
class Config:
    """Main configuration container."""
    
    api: APIConfig = field(default_factory=APIConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# Global configuration instance
config = Config()


def validate_config() -> list[str]:
    """
    Validate configuration and return list of issues.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []
    
    if not config.api.api_key:
        issues.append("OPENROUTER_API_KEY environment variable not set")
    
    if not 0.0 <= config.evaluation.semantic_threshold <= 1.0:
        issues.append(f"Invalid semantic_threshold: {config.evaluation.semantic_threshold}")
    
    if config.evaluation.samples_per_config < 1:
        issues.append("samples_per_config must be at least 1")
    
    return issues
