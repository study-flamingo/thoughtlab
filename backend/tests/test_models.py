import pytest
from pydantic import ValidationError
from app.models.nodes import (
    ObservationCreate,
    SourceCreate,
    HypothesisCreate,
    RelationshipCreate,
)


def test_observation_create_valid():
    """Test valid observation creation"""
    obs = ObservationCreate(
        text="Test observation",
        confidence=0.8,
        concept_names=["test"],
    )
    assert obs.text == "Test observation"
    assert obs.confidence == 0.8
    assert obs.concept_names == ["test"]


def test_observation_create_defaults():
    """Test observation with defaults"""
    obs = ObservationCreate(text="Test")
    assert obs.confidence == 0.8
    assert obs.concept_names is None


def test_observation_create_validation():
    """Test observation validation"""
    # Empty text should fail
    with pytest.raises(ValidationError):
        ObservationCreate(text="")

    # Invalid confidence should fail
    with pytest.raises(ValidationError):
        ObservationCreate(text="Test", confidence=1.5)

    with pytest.raises(ValidationError):
        ObservationCreate(text="Test", confidence=-0.1)


def test_source_create_valid():
    """Test valid source creation"""
    source = SourceCreate(
        title="Test Source",
        url="https://example.com",
        source_type="paper",
    )
    assert source.title == "Test Source"
    assert source.url == "https://example.com"
    assert source.source_type == "paper"


def test_hypothesis_create_valid():
    """Test valid hypothesis creation"""
    hypothesis = HypothesisCreate(
        name="Test Name",
        claim="Test hypothesis",
        status="proposed",
    )
    assert hypothesis.name == "Test Name"
    assert hypothesis.claim == "Test hypothesis"
    assert hypothesis.status == "proposed"


def test_relationship_create_valid():
    """Test valid relationship creation"""
    rel = RelationshipCreate(
        from_id="id1",
        to_id="id2",
        relationship_type="SUPPORTS",
        confidence=0.9,
    )
    assert rel.from_id == "id1"
    assert rel.to_id == "id2"
    assert rel.relationship_type == "SUPPORTS"
    assert rel.confidence == 0.9
