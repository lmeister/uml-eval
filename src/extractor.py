"""
Triple Extractor for UML Class Diagrams.

This module transforms parsed UML class diagrams into atomic (Subject, Predicate, Object)
triples following the schema defined in the thesis (Section 4.2.1).

Triple Schema:
- Entity:       (ClassName, isClass, _)
- Attribute:    (ClassName, hasAttribute, AttributeName)
- Method:       (ClassName, hasMethod, MethodName)
- Association:  (ClassA, associatesWith, ClassB) - alphabetized for undirected
- Inheritance:  (Child, inheritsFrom, Parent) - direction preserved
- Composition:  (Part, partOf, Whole) - direction preserved
- Aggregation:  (Part, aggregatedIn, Whole) - direction preserved

The hybrid directionality approach treats associations as undirected (semantically
bidirectional) while preserving direction for inheritance, composition, and aggregation
(which have inherent semantic direction).
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

from parser import ParsedDiagram, RelationshipType


class PredicateType(Enum):
    """Predicate types for our triple schema."""
    IS_CLASS = "isClass"
    HAS_ATTRIBUTE = "hasAttribute"
    HAS_METHOD = "hasMethod"
    ASSOCIATES_WITH = "associatesWith"
    INHERITS_FROM = "inheritsFrom"
    PART_OF = "partOf"
    AGGREGATED_IN = "aggregatedIn"


@dataclass(frozen=True)
class Triple:
    """
    Represents an atomic semantic unit from a UML diagram.
    
    Frozen dataclass enables use in sets for comparison operations.
    """
    subject: str
    predicate: PredicateType
    obj: Optional[str]  # None for class existence triples
    
    def __str__(self) -> str:
        obj_str = self.obj if self.obj else "_"
        return f"({self.subject}, {self.predicate.value}, {obj_str})"
    
    def __hash__(self) -> int:
        return hash((self.subject, self.predicate, self.obj))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Triple):
            return False
        return (self.subject == other.subject and 
                self.predicate == other.predicate and 
                self.obj == other.obj)
    
    def to_tuple(self) -> tuple:
        """Convert to tuple for serialization."""
        return (self.subject, self.predicate.value, self.obj)
    
    @classmethod
    def from_tuple(cls, t: tuple) -> 'Triple':
        """Create from tuple."""
        return cls(
            subject=t[0],
            predicate=PredicateType(t[1]),
            obj=t[2]
        )


@dataclass
class TripleSet:
    """
    A set of triples extracted from a diagram.
    
    Provides convenient access to triples by category and
    methods for comparison operations.
    """
    triples: set[Triple]
    
    def __init__(self, triples: Optional[set[Triple]] = None):
        self.triples = triples if triples else set()
    
    def add(self, triple: Triple) -> None:
        """Add a triple to the set."""
        self.triples.add(triple)
    
    def __len__(self) -> int:
        return len(self.triples)
    
    def __iter__(self):
        return iter(self.triples)
    
    def __and__(self, other: 'TripleSet') -> 'TripleSet':
        """Set intersection."""
        return TripleSet(self.triples & other.triples)
    
    def __or__(self, other: 'TripleSet') -> 'TripleSet':
        """Set union."""
        return TripleSet(self.triples | other.triples)
    
    def __sub__(self, other: 'TripleSet') -> 'TripleSet':
        """Set difference."""
        return TripleSet(self.triples - other.triples)
    
    @property
    def class_triples(self) -> set[Triple]:
        """Get all class existence triples."""
        return {t for t in self.triples if t.predicate == PredicateType.IS_CLASS}
    
    @property
    def attribute_triples(self) -> set[Triple]:
        """Get all attribute triples."""
        return {t for t in self.triples if t.predicate == PredicateType.HAS_ATTRIBUTE}
    
    @property
    def method_triples(self) -> set[Triple]:
        """Get all method triples."""
        return {t for t in self.triples if t.predicate == PredicateType.HAS_METHOD}
    
    @property
    def relationship_triples(self) -> set[Triple]:
        """Get all relationship triples."""
        rel_predicates = {
            PredicateType.ASSOCIATES_WITH,
            PredicateType.INHERITS_FROM,
            PredicateType.PART_OF,
            PredicateType.AGGREGATED_IN
        }
        return {t for t in self.triples if t.predicate in rel_predicates}
    
    def get_all_names(self) -> set[str]:
        """Extract all unique names (classes, attributes, methods) from triples."""
        names = set()
        for triple in self.triples:
            names.add(triple.subject)
            if triple.obj:
                names.add(triple.obj)
        return names
    
    def to_list(self) -> list[tuple]:
        """Convert to list of tuples for serialization."""
        return [t.to_tuple() for t in self.triples]
    
    @classmethod
    def from_list(cls, data: list[tuple]) -> 'TripleSet':
        """Create from list of tuples."""
        return cls({Triple.from_tuple(t) for t in data})


class TripleExtractor:
    """
    Extracts triples from parsed UML class diagrams.
    
    Implements the hybrid directionality approach:
    - Associations are treated as undirected (alphabetized class names)
    - Inheritance, composition, aggregation preserve direction
    """
    
    def extract(self, diagram: ParsedDiagram) -> TripleSet:
        """
        Extract all triples from a parsed diagram.
        
        Args:
            diagram: ParsedDiagram from the parser
            
        Returns:
            TripleSet containing all extracted triples
        """
        result = TripleSet()
        
        # Extract class triples
        for cls in diagram.classes:
            result.add(Triple(
                subject=cls.name,
                predicate=PredicateType.IS_CLASS,
                obj=None
            ))
            
            # Extract attribute triples
            for attr in cls.attributes:
                result.add(Triple(
                    subject=cls.name,
                    predicate=PredicateType.HAS_ATTRIBUTE,
                    obj=attr.name
                ))
            
            # Extract method triples
            for method in cls.methods:
                result.add(Triple(
                    subject=cls.name,
                    predicate=PredicateType.HAS_METHOD,
                    obj=method.name
                ))
        
        # Extract relationship triples
        for rel in diagram.relationships:
            result.add(self._extract_relationship_triple(rel))
        
        return result
    
    def _extract_relationship_triple(self, rel) -> Triple:
        """
        Extract a triple from a relationship.
        
        Handles the hybrid directionality:
        - Associations: alphabetize class names (undirected)
        - Others: preserve semantic direction
        """
        if rel.rel_type == RelationshipType.ASSOCIATION:
            # Alphabetize for undirected semantics
            names = sorted([rel.source, rel.target])
            return Triple(
                subject=names[0],
                predicate=PredicateType.ASSOCIATES_WITH,
                obj=names[1]
            )
        
        elif rel.rel_type == RelationshipType.INHERITANCE:
            # source = Child, target = Parent
            return Triple(
                subject=rel.source,
                predicate=PredicateType.INHERITS_FROM,
                obj=rel.target
            )
        
        elif rel.rel_type == RelationshipType.COMPOSITION:
            # In PlantUML: Part *-- Whole
            # We represent as (Part, partOf, Whole)
            return Triple(
                subject=rel.source,
                predicate=PredicateType.PART_OF,
                obj=rel.target
            )
        
        elif rel.rel_type == RelationshipType.AGGREGATION:
            # Similar to composition
            return Triple(
                subject=rel.source,
                predicate=PredicateType.AGGREGATED_IN,
                obj=rel.target
            )
        
        else:
            raise ValueError(f"Unknown relationship type: {rel.rel_type}")


def extract_triples(diagram: ParsedDiagram) -> TripleSet:
    """Convenience function to extract triples using default extractor."""
    extractor = TripleExtractor()
    return extractor.extract(diagram)


if __name__ == "__main__":
    # Test with example
    from parser import parse_plantuml
    
    test_diagram = """
    @startuml
    class Person {
        name : String
        age : int
        getName()
    }
    
    class Student {
        studentId : String
    }
    
    class Course {
        title : String
    }
    
    Student --|> Person
    Student "0..*" --> "1..*" Course : enrolledIn
    @enduml
    """
    
    parsed = parse_plantuml(test_diagram)
    triples = extract_triples(parsed)
    
    print(f"Extracted {len(triples)} triples:\n")
    
    print("Class triples:")
    for t in triples.class_triples:
        print(f"  {t}")
    
    print("\nAttribute triples:")
    for t in triples.attribute_triples:
        print(f"  {t}")
    
    print("\nMethod triples:")
    for t in triples.method_triples:
        print(f"  {t}")
    
    print("\nRelationship triples:")
    for t in triples.relationship_triples:
        print(f"  {t}")
    
    print(f"\nAll names in diagram: {triples.get_all_names()}")
