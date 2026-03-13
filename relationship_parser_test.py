#!/usr/bin/env python3
"""
Comprehensive tests for the relationship parser.
"""

from relationship_parser import RelationshipParser


def test_basic_parsing():
    """Test basic parsing functionality."""
    parser = RelationshipParser()
    
    # Test single line parsing
    line = "| Stacey Monroe -> Supports -> Approval of Tess O'Brien Apartments |"
    result = parser.parse_single(line)
    
    assert result.entity == "Stacey Monroe"
    assert result.relationship == "Supports"
    assert result.issue == "Approval of Tess O'Brien Apartments"
    
    print("✓ Basic parsing works")


def test_multiple_parsing():
    """Test parsing multiple relationships."""
    parser = RelationshipParser()
    
    test_input = """
| Stacey Monroe -> Supports -> Approval of Tess O'Brien Apartments |
| John Smith -> Opposes -> Building permit for new construction |
| City Council -> Votes -> On zoning changes |
"""
    
    relationships = parser.parse_multiple(test_input)
    
    assert len(relationships) == 3
    assert relationships[0].entity == "Stacey Monroe"
    assert relationships[0].relationship == "Supports"
    assert relationships[0].issue == "Approval of Tess O'Brien Apartments"
    
    assert relationships[1].entity == "John Smith"
    assert relationships[1].relationship == "Opposes"
    assert relationships[1].issue == "Building permit for new construction"
    
    assert relationships[2].entity == "City Council"
    assert relationships[2].relationship == "Votes"
    assert relationships[2].issue == "On zoning changes"
    
    print("✓ Multiple parsing works")


def test_edge_cases():
    """Test edge cases and whitespace handling."""
    parser = RelationshipParser()
    
    # Test with extra whitespace
    line = " |   Stacey Monroe   ->   Supports   ->   Approval of Tess O'Brien Apartments   | "
    result = parser.parse_single(line)
    
    assert result.entity == "Stacey Monroe"
    assert result.relationship == "Supports"
    assert result.issue == "Approval of Tess O'Brien Apartments"
    
    print("✓ Edge cases handled correctly")


def test_missing_bars():
    """Test that lines without | bars still parse."""
    parser = RelationshipParser()

    line = "Stacey Monroe -> Supports -> Approval of Tess O'Brien Apartments"
    result = parser.parse_single(line)
    assert result.entity == "Stacey Monroe"
    assert result.relationship == "Supports"
    assert result.issue == "Approval of Tess O'Brien Apartments"

    # Missing only the trailing bar
    line = "| John Smith -> Opposes -> Building permit for new construction"
    result = parser.parse_single(line)
    assert result.entity == "John Smith"
    assert result.relationship == "Opposes"
    assert result.issue == "Building permit for new construction"

    print("✓ Missing bars handled correctly")


def test_complex_issues():
    """Test parsing with complex issue descriptions."""
    parser = RelationshipParser()
    
    # Test with issues containing dashes, colons, etc.
    line = "| Committee -> Reviews -> Annual budget proposal for 2024: Q1 spending |"
    result = parser.parse_single(line)
    
    assert result.entity == "Committee"
    assert result.relationship == "Reviews"
    assert result.issue == "Annual budget proposal for 2024: Q1 spending"
    
    print("✓ Complex issues parsed correctly")


if __name__ == "__main__":
    test_basic_parsing()
    test_multiple_parsing()
    test_edge_cases()
    test_complex_issues()
    print("All tests passed! 🎉")