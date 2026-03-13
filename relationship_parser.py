#!/usr/bin/env python3
"""
Parser for informal text-graph format:
| person-or-entity -> relationship-type -> issue
"""

import re
from typing import NamedTuple


class Relationship(NamedTuple):
    """Represents a parsed relationship from the text format."""
    entity: str
    relationship: str
    issue: str


class RelationshipParser:
    """Parses informal text-graph format into structured relationships."""
    
    def __init__(self):
        # Match x -> y -> z with optional surrounding | bars
        self.pattern = re.compile(
            r'^\|?\s*(.*?)\s*->\s*(.*?)\s*->\s*(.*?)\s*\|?$',
            re.MULTILINE
        )
    
    def parse_single(self, line: str) -> Relationship:
        """
        Parse a single line in the format:
        | entity -> relationship -> issue
        
        Args:
            line: The input line to parse
            
        Returns:
            Relationship object with entity, relationship, and issue
            
        Raises:
            ValueError: If the line doesn't match the expected format
        """
        match = self.pattern.match(line.strip())
        if not match:
            raise ValueError(f"Line does not match expected format: {line}")
        
        entity, relationship, issue = match.groups()
        return Relationship(entity.strip(), relationship.strip(), issue.strip())
    
    def parse_multiple(self, text: str) -> list[Relationship]:
        """
        Parse multiple lines of the format into a list of relationships.
        
        Args:
            text: Text containing one or more lines in the format
            
        Returns:
            List of Relationship objects
        """
        relationships = []
        lines = text.strip().split('\n')
        
        for line in lines:
            if line.strip():
                try:
                    relationships.append(self.parse_single(line))
                except ValueError:
                    pass
        
        return relationships


# Example usage
if __name__ == "__main__":
    # Test data
    test_input = """
| Stacey Monroe -> Supports -> Approval of Tess O'Brien Apartments |
| John Smith -> Opposes -> Building permit for new construction |
| City Council -> Votes -> On zoning changes |
"""
    
    parser = RelationshipParser()
    relationships = parser.parse_multiple(test_input)
    
    for rel in relationships:
        print(f"Entity: {rel.entity}")
        print(f"Relationship: {rel.relationship}")
        print(f"Issue: {rel.issue}")
        print("---")