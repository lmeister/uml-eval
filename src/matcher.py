"""
Semantic Matcher for UML Diagram Evaluation.

This module implements the reference-based evaluation strategy using:
1. SBERT embeddings for semantic similarity between names
2. Cosine similarity with configurable threshold
3. Triple normalization based on name mappings
4. Precision, recall, and F1 computation

The semantic matching allows recognition of equivalent names like
"Customer" <-> "Client" or "OrderItem" <-> "LineItem" that would
fail exact string matching.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from sentence_transformers import SentenceTransformer

from extractor import Triple, TripleSet, PredicateType
from config import config


@dataclass
class MatchingResult:
    """Results of semantic matching between two triple sets."""
    
    # Name mappings: generated_name -> reference_name
    name_mappings: dict[str, str]
    
    # Similarity scores for each mapping
    similarity_scores: dict[str, float]
    
    # Triple sets after normalization
    normalized_generated: TripleSet
    reference: TripleSet
    
    # Set operations results
    matched_triples: TripleSet      # Intersection
    missing_triples: TripleSet      # In reference but not generated
    excess_triples: TripleSet       # In generated but not reference
    
    # Metrics
    precision: float
    recall: float
    f1_score: float
    
    def __str__(self) -> str:
        return (
            f"MatchingResult(\n"
            f"  precision={self.precision:.3f},\n"
            f"  recall={self.recall:.3f},\n"
            f"  f1={self.f1_score:.3f},\n"
            f"  matched={len(self.matched_triples)},\n"
            f"  missing={len(self.missing_triples)},\n"
            f"  excess={len(self.excess_triples)}\n"
            f")"
        )


class SemanticMatcher:
    """
    Performs semantic matching between generated and reference UML diagrams.
    
    Uses SBERT embeddings to compute similarity between names, allowing
    semantically equivalent but lexically different names to match.
    """
    
    def __init__(
        self,
        model_name: str = None,
        threshold: float = None
    ):
        """
        Initialize the semantic matcher.
        
        Args:
            model_name: SBERT model to use (default from config)
            threshold: Cosine similarity threshold for matching (default from config)
        """
        self.model_name = model_name or config.evaluation.sbert_model
        self.threshold = threshold if threshold is not None else config.evaluation.semantic_threshold
        
        # Lazy load model
        self._model = None
        
        # Cache for embeddings
        self._embedding_cache: dict[str, np.ndarray] = {}
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the SBERT model."""
        if self._model is None:
            print(f"Loading SBERT model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text string, using cache.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if text not in self._embedding_cache:
            # Handle camelCase and compound names by adding spaces
            processed = self._preprocess_name(text)
            self._embedding_cache[text] = self.model.encode(processed)
        return self._embedding_cache[text]
    
    def _preprocess_name(self, name: str) -> str:
        """
        Preprocess a name for better embedding.
        
        Handles CamelCase by inserting spaces: "OrderLineItem" -> "Order Line Item"
        This improves embedding quality for compound names.
        """
        import re
        # Insert space before uppercase letters (but not at start)
        processed = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
        return processed.lower()
    
    def compute_similarity(self, name1: str, name2: str) -> float:
        """
        Compute cosine similarity between two names.
        
        Args:
            name1: First name
            name2: Second name
            
        Returns:
            Cosine similarity in range [-1, 1]
        """
        emb1 = self.get_embedding(name1)
        emb2 = self.get_embedding(name2)
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_best_mapping(
        self,
        generated_names: set[str],
        reference_names: set[str]
    ) -> tuple[dict[str, str], dict[str, float]]:
        """
        Find optimal name mappings from generated to reference names.
        
        For each generated name, finds the reference name with highest
        similarity above the threshold.
        
        Args:
            generated_names: Names from generated diagram
            reference_names: Names from reference diagram
            
        Returns:
            Tuple of (mapping dict, similarity scores dict)
        """
        mappings = {}
        scores = {}
        
        for gen_name in generated_names:
            best_match = None
            # Use a very low float to catch any similarity score
            best_score = float('-inf') 
            
            for ref_name in reference_names:
                sim = self.compute_similarity(gen_name, ref_name)
                if sim > best_score:
                    best_score = sim
                    best_match = ref_name
            
            # Check if a match was found AND it meets the threshold
            # If threshold is 0.0, any best_match found will be used even if score is negative
            if best_match is not None and (best_score >= self.threshold or self.threshold <= 0.0):
                mappings[gen_name] = best_match
                scores[gen_name] = best_score
            else:
                # Fallback to original name (hallucination indicator)
                mappings[gen_name] = gen_name
                scores[gen_name] = 0.0
                
        return mappings, scores
    
    def normalize_triples(
        self,
        triples: TripleSet,
        name_mappings: dict[str, str]
    ) -> TripleSet:
        """
        Normalize triples by applying name mappings.
        
        Replaces names in triples with their mapped equivalents
        to enable set-based comparison.
        
        Args:
            triples: Original triple set
            name_mappings: Mapping from original to normalized names
            
        Returns:
            New TripleSet with normalized names
        """
        normalized = TripleSet()
        
        for triple in triples:
            # Map subject
            new_subject = name_mappings.get(triple.subject, triple.subject)
            
            # Map object (if present)
            new_obj = None
            if triple.obj:
                new_obj = name_mappings.get(triple.obj, triple.obj)
            
            normalized.add(Triple(
                subject=new_subject,
                predicate=triple.predicate,
                obj=new_obj
            ))
        
        return normalized
    
    def match(
        self,
        generated: TripleSet,
        reference: TripleSet
    ) -> MatchingResult:
        """
        Perform complete matching between generated and reference triple sets.
        
        Args:
            generated: Triples from generated diagram
            reference: Triples from reference diagram
            
        Returns:
            MatchingResult with metrics and analysis
        """
        # Extract all names from both sets
        gen_names = generated.get_all_names()
        ref_names = reference.get_all_names()
        
        # Find best mappings
        mappings, scores = self.find_best_mapping(gen_names, ref_names)
        
        # Normalize generated triples
        normalized_gen = self.normalize_triples(generated, mappings)
        
        # Compute set operations
        matched = normalized_gen & reference
        missing = reference - normalized_gen
        excess = normalized_gen - reference
        
        # Compute metrics
        precision = len(matched) / len(normalized_gen) if len(normalized_gen) > 0 else 0.0
        recall = len(matched) / len(reference) if len(reference) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return MatchingResult(
            name_mappings=mappings,
            similarity_scores=scores,
            normalized_generated=normalized_gen,
            reference=reference,
            matched_triples=matched,
            missing_triples=missing,
            excess_triples=excess,
            precision=precision,
            recall=recall,
            f1_score=f1
        )


def create_matcher(threshold: float = None) -> SemanticMatcher:
    """Create a semantic matcher with optional custom threshold."""
    return SemanticMatcher(threshold=threshold)


def evaluate_diagram(
    generated: TripleSet,
    reference: TripleSet,
    threshold: float = None
) -> MatchingResult:
    """
    Convenience function to evaluate a generated diagram against reference.
    
    Args:
        generated: Triples from generated diagram
        reference: Triples from reference diagram
        threshold: Optional custom similarity threshold
        
    Returns:
        MatchingResult with complete evaluation
    """
    matcher = create_matcher(threshold=threshold)
    return matcher.match(generated, reference)


if __name__ == "__main__":
    # Test with example diagrams showing semantic matching
    from parser import parse_plantuml
    from extractor import extract_triples
    
    # Reference diagram uses "Customer"
    reference_plantuml = """
    @startuml
    class Customer {
        name : String
        email : String
    }
    
    class Order {
        orderId : int
        totalAmount : double
    }
    
    Customer "1" --> "0..*" Order : places
    @enduml
    """
    
    # Generated diagram uses "Client" instead (should match semantically)
    generated_plantuml = """
    @startuml
    class Client {
        name : String
        emailAddress : String
    }
    
    class Order {
        orderId : int
        total : double
    }
    
    class Invoice {
        invoiceNumber : String
    }
    
    Client "1" --> "0..*" Order : places
    Order --> Invoice
    @enduml
    """
    
    # Parse and extract
    ref_parsed = parse_plantuml(reference_plantuml)
    gen_parsed = parse_plantuml(generated_plantuml)
    
    ref_triples = extract_triples(ref_parsed)
    gen_triples = extract_triples(gen_parsed)
    
    print(f"Reference triples: {len(ref_triples)}")
    print(f"Generated triples: {len(gen_triples)}")
    
    # Match
    matcher = SemanticMatcher()
    result = matcher.match(gen_triples, ref_triples)
    
    print(f"\n{result}")
    
    print("\nName Mappings:")
    for gen, ref in result.name_mappings.items():
        score = result.similarity_scores.get(gen, 0.0)
        print(f"  {gen} -> {ref} (sim={score:.3f})")
    
    print("\nMatched Triples:")
    for t in result.matched_triples:
        print(f"  {t}")
    
    print("\nMissing Triples (in reference, not generated):")
    for t in result.missing_triples:
        print(f"  {t}")
    
    print("\nExcess Triples (in generated, not reference - potential hallucination):")
    for t in result.excess_triples:
        print(f"  {t}")
