"""
PlantUML Parser for Constrained UML Class Diagrams.

This module parses PlantUML syntax conforming to our constrained format
and extracts structured representations of classes, attributes, methods,
and relationships.

The constrained format accepts:
- Class declarations with attributes and methods
- Four relationship types: association, inheritance, composition, aggregation
- Multiplicities on associations

It excludes: packages, notes, visibility modifiers, stereotypes, interfaces
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class RelationshipType(Enum):
    """Types of UML relationships we support."""
    ASSOCIATION = "association"
    INHERITANCE = "inheritance"
    COMPOSITION = "composition"
    AGGREGATION = "aggregation"


@dataclass
class Attribute:
    """Represents a class attribute."""
    name: str
    type: Optional[str] = None
    
    def __str__(self) -> str:
        if self.type:
            return f"{self.name} : {self.type}"
        return self.name


@dataclass
class Method:
    """Represents a class method (name only, no parameters per our constraint)."""
    name: str
    
    def __str__(self) -> str:
        return f"{self.name}()"


@dataclass
class UMLClass:
    """Represents a UML class with its members."""
    name: str
    attributes: list[Attribute] = field(default_factory=list)
    methods: list[Method] = field(default_factory=list)
    
    def __str__(self) -> str:
        lines = [f"class {self.name} {{"]
        for attr in self.attributes:
            lines.append(f"  {attr}")
        for method in self.methods:
            lines.append(f"  {method}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class Relationship:
    """Represents a relationship between two classes."""
    source: str
    target: str
    rel_type: RelationshipType
    source_multiplicity: Optional[str] = None
    target_multiplicity: Optional[str] = None
    label: Optional[str] = None
    
    def __str__(self) -> str:
        symbols = {
            RelationshipType.ASSOCIATION: "-->",
            RelationshipType.INHERITANCE: "--|>",
            RelationshipType.COMPOSITION: "*--",
            RelationshipType.AGGREGATION: "o--"
        }
        return f"{self.source} {symbols[self.rel_type]} {self.target}"


@dataclass
class ParsedDiagram:
    """Complete parsed representation of a UML class diagram."""
    classes: list[UMLClass] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    raw_text: str = ""
    parse_errors: list[str] = field(default_factory=list)
    
    def get_class(self, name: str) -> Optional[UMLClass]:
        """Find a class by name."""
        for cls in self.classes:
            if cls.name == name:
                return cls
        return None
    
    def __str__(self) -> str:
        lines = ["@startuml"]
        for cls in self.classes:
            lines.append(str(cls))
            lines.append("")
        for rel in self.relationships:
            lines.append(str(rel))
        lines.append("@enduml")
        return "\n".join(lines)


class PlantUMLParser:
    """
    Parser for constrained PlantUML class diagram syntax.
    
    Uses regex-based extraction which is sufficient for our simplified
    format. For full PlantUML support, a grammar-based parser would be needed.
    """
    
    # Regex patterns for parsing
    PATTERNS = {
        # Match class declarations: class ClassName { ... }
        'class_block': re.compile(
            r'class\s+(\w+)\s*\{([^}]*)\}',
            re.MULTILINE | re.DOTALL
        ),
        
        # Match empty class declarations: class ClassName
        'class_empty': re.compile(
            r'^\s*class\s+(\w+)\s*$',
            re.MULTILINE
        ),
        
        # Match attribute: attributeName : Type
        'attribute': re.compile(
            r'^\s*(\w+)\s*:\s*(\w+(?:\s*<[^>]+>)?)\s*$',
            re.MULTILINE
        ),
        
        # Match method: methodName()
        'method': re.compile(
            r'^\s*(\w+)\s*\(\s*\)\s*$',
            re.MULTILINE
        ),
        
        # Association: ClassA "mult" --> "mult" ClassB : label
        'association': re.compile(
            r'(\w+)\s*(?:"([^"]*)")?\s*-->\s*(?:"([^"]*)")?\s*(\w+)(?:\s*:\s*(\w+))?'
        ),
        
        # Inheritance: Child --|> Parent
        'inheritance': re.compile(
            r'(\w+)\s*--\|>\s*(\w+)'
        ),
        
        # Composition: Part *-- Whole or Whole *-- Part
        'composition': re.compile(
            r'(\w+)\s*(?:"([^"]*)")?\s*\*--\s*(?:"([^"]*)")?\s*(\w+)(?:\s*:\s*(\w+))?'
        ),
        
        # Aggregation: Part o-- Whole or Whole o-- Part  
        'aggregation': re.compile(
            r'(\w+)\s*(?:"([^"]*)")?\s*o--\s*(?:"([^"]*)")?\s*(\w+)(?:\s*:\s*(\w+))?'
        ),
    }
    
    def parse(self, plantuml_text: str) -> ParsedDiagram:
        """
        Parse PlantUML text into structured representation.
        
        Args:
            plantuml_text: PlantUML diagram as string
            
        Returns:
            ParsedDiagram with extracted classes and relationships
        """
        result = ParsedDiagram(raw_text=plantuml_text)
        
        # Extract content between @startuml and @enduml
        content = self._extract_diagram_content(plantuml_text)
        if content is None:
            result.parse_errors.append("Could not find @startuml/@enduml markers")
            # Try parsing anyway - LLM might have omitted markers
            content = plantuml_text
        
        # Parse classes
        self._parse_classes(content, result)
        
        # Parse relationships
        self._parse_relationships(content, result)
        
        return result
    
    def _extract_diagram_content(self, text: str) -> Optional[str]:
        """Extract content between @startuml and @enduml markers."""
        # Handle potential code block markers from LLM output
        text = re.sub(r'```(?:plantuml|uml)?\n?', '', text)
        text = re.sub(r'```\n?', '', text)
        
        match = re.search(
            r'@startuml\s*(.*?)\s*@enduml',
            text,
            re.DOTALL | re.IGNORECASE
        )
        return match.group(1) if match else None
    
    def _parse_classes(self, content: str, result: ParsedDiagram) -> None:
        """Parse all class declarations from content."""
        # Parse classes with body
        for match in self.PATTERNS['class_block'].finditer(content):
            class_name = match.group(1)
            class_body = match.group(2)
            
            uml_class = UMLClass(name=class_name)
            
            # Parse attributes
            for attr_match in self.PATTERNS['attribute'].finditer(class_body):
                uml_class.attributes.append(Attribute(
                    name=attr_match.group(1),
                    type=attr_match.group(2)
                ))
            
            # Parse methods
            for method_match in self.PATTERNS['method'].finditer(class_body):
                uml_class.methods.append(Method(name=method_match.group(1)))
            
            result.classes.append(uml_class)
        
        # Parse empty classes (no body)
        for match in self.PATTERNS['class_empty'].finditer(content):
            class_name = match.group(1)
            # Check if not already parsed as block class
            if not result.get_class(class_name):
                result.classes.append(UMLClass(name=class_name))
    
    def _parse_relationships(self, content: str, result: ParsedDiagram) -> None:
        """Parse all relationship declarations from content."""
        # Parse inheritance (must check before associations due to overlap)
        for match in self.PATTERNS['inheritance'].finditer(content):
            result.relationships.append(Relationship(
                source=match.group(1),  # Child
                target=match.group(2),  # Parent
                rel_type=RelationshipType.INHERITANCE
            ))
        
        # Parse composition
        for match in self.PATTERNS['composition'].finditer(content):
            result.relationships.append(Relationship(
                source=match.group(1),
                target=match.group(4),
                rel_type=RelationshipType.COMPOSITION,
                source_multiplicity=match.group(2),
                target_multiplicity=match.group(3),
                label=match.group(5) if len(match.groups()) > 4 else None
            ))
        
        # Parse aggregation
        for match in self.PATTERNS['aggregation'].finditer(content):
            result.relationships.append(Relationship(
                source=match.group(1),
                target=match.group(4),
                rel_type=RelationshipType.AGGREGATION,
                source_multiplicity=match.group(2),
                target_multiplicity=match.group(3),
                label=match.group(5) if len(match.groups()) > 4 else None
            ))
        
        # Parse associations (check that it's not inheritance/comp/agg)
        for match in self.PATTERNS['association'].finditer(content):
            source = match.group(1)
            target = match.group(4)
            
            # Skip if this looks like it was already parsed as another type
            # (regex might have partial matches)
            line_text = match.group(0)
            if '--|>' in line_text or '*--' in line_text or 'o--' in line_text:
                continue
            
            result.relationships.append(Relationship(
                source=source,
                target=target,
                rel_type=RelationshipType.ASSOCIATION,
                source_multiplicity=match.group(2),
                target_multiplicity=match.group(3),
                label=match.group(5) if len(match.groups()) > 4 else None
            ))


def extract_plantuml_from_response(llm_response: str) -> str:
    """
    Extract PlantUML code block from LLM response.
    
    LLMs sometimes include explanatory text even when asked not to.
    This function extracts just the PlantUML diagram.
    
    Args:
        llm_response: Full text response from LLM
        
    Returns:
        Extracted PlantUML code (or original if markers found)
    """
    # Try to find @startuml...@enduml block
    match = re.search(
        r'@startuml.*?@enduml',
        llm_response,
        re.DOTALL | re.IGNORECASE
    )
    
    if match:
        return match.group(0)
    
    # Try to find code block
    code_match = re.search(
        r'```(?:plantuml|uml)?\s*(.*?)```',
        llm_response,
        re.DOTALL
    )
    
    if code_match:
        return code_match.group(1).strip()
    
    # Return original - parser will try to handle it
    return llm_response


# Convenience function
def parse_plantuml(text: str) -> ParsedDiagram:
    """Parse PlantUML text using default parser."""
    parser = PlantUMLParser()
    return parser.parse(text)


if __name__ == "__main__":
    # Test with example
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
    
    result = parse_plantuml(test_diagram)
    print("Parsed Classes:")
    for cls in result.classes:
        print(f"  {cls.name}: {len(cls.attributes)} attrs, {len(cls.methods)} methods")
    
    print("\nParsed Relationships:")
    for rel in result.relationships:
        print(f"  {rel.source} --[{rel.rel_type.value}]--> {rel.target}")
