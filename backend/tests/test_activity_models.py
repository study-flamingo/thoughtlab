"""Tests for Activity Feed models."""

import pytest
from pydantic import ValidationError
from datetime import datetime, UTC
from app.models.activity import (
    ActivityType,
    ActivityStatus,
    SuggestionData,
    ProcessingData,
    ActivityCreate,
    ActivityUpdate,
    ActivityResponse,
    ActivityFilter,
    SuggestionThresholds,
    DEFAULT_THRESHOLDS,
)


class TestActivityType:
    """Tests for ActivityType enum."""

    def test_all_expected_types_exist(self):
        """Verify all expected activity types are defined."""
        expected = [
            "node_created",
            "node_updated",
            "node_deleted",
            "relationship_created",
            "relationship_updated",
            "relationship_deleted",
            "relationship_suggested",
            "relationship_auto_created",
            "processing_started",
            "processing_chunking",
            "processing_embedding",
            "processing_analyzing",
            "processing_completed",
            "processing_failed",
            "error",
            "warning",
            "info",
        ]
        actual = [t.value for t in ActivityType]
        assert set(expected) == set(actual)

    def test_activity_type_is_string(self):
        """Verify ActivityType values are strings for JSON serialization."""
        assert ActivityType.NODE_CREATED.value == "node_created"
        assert isinstance(ActivityType.NODE_CREATED.value, str)


class TestActivityStatus:
    """Tests for ActivityStatus enum."""

    def test_all_expected_statuses_exist(self):
        """Verify all expected statuses are defined."""
        expected = ["pending", "approved", "rejected", "expired"]
        actual = [s.value for s in ActivityStatus]
        assert set(expected) == set(actual)


class TestSuggestionData:
    """Tests for SuggestionData model."""

    def test_valid_suggestion_data(self):
        """Test creating valid suggestion data."""
        data = SuggestionData(
            from_node_id="node-1",
            from_node_type="Observation",
            from_node_label="First observation",
            to_node_id="node-2",
            to_node_type="Hypothesis",
            to_node_label="A hypothesis",
            relationship_type="SUPPORTS",
            confidence=0.85,
            reasoning="These concepts are closely related",
        )
        assert data.from_node_id == "node-1"
        assert data.confidence == 0.85
        assert data.reasoning == "These concepts are closely related"

    def test_suggestion_data_without_reasoning(self):
        """Test suggestion data without optional reasoning."""
        data = SuggestionData(
            from_node_id="node-1",
            from_node_type="Observation",
            from_node_label="First",
            to_node_id="node-2",
            to_node_type="Hypothesis",
            to_node_label="Second",
            relationship_type="RELATES_TO",
            confidence=0.7,
        )
        assert data.reasoning is None

    def test_suggestion_data_confidence_validation(self):
        """Test confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            SuggestionData(
                from_node_id="node-1",
                from_node_type="Observation",
                from_node_label="First",
                to_node_id="node-2",
                to_node_type="Hypothesis",
                to_node_label="Second",
                relationship_type="RELATES_TO",
                confidence=1.5,  # Invalid: > 1
            )

        with pytest.raises(ValidationError):
            SuggestionData(
                from_node_id="node-1",
                from_node_type="Observation",
                from_node_label="First",
                to_node_id="node-2",
                to_node_type="Hypothesis",
                to_node_label="Second",
                relationship_type="RELATES_TO",
                confidence=-0.1,  # Invalid: < 0
            )


class TestProcessingData:
    """Tests for ProcessingData model."""

    def test_valid_processing_data(self):
        """Test creating valid processing data."""
        data = ProcessingData(
            node_id="node-1",
            node_type="Source",
            node_label="Research Paper",
            stage="embedding",
            progress=0.5,
            chunks_created=10,
            embeddings_created=5,
        )
        assert data.node_id == "node-1"
        assert data.stage == "embedding"
        assert data.progress == 0.5
        assert data.chunks_created == 10

    def test_processing_data_with_error(self):
        """Test processing data with error message."""
        data = ProcessingData(
            node_id="node-1",
            node_type="Source",
            node_label="Failed Paper",
            stage="failed",
            error_message="API rate limit exceeded",
        )
        assert data.stage == "failed"
        assert data.error_message == "API rate limit exceeded"

    def test_processing_data_progress_validation(self):
        """Test progress must be between 0 and 1."""
        with pytest.raises(ValidationError):
            ProcessingData(
                node_id="node-1",
                node_type="Source",
                node_label="Test",
                stage="embedding",
                progress=1.5,  # Invalid
            )


class TestActivityCreate:
    """Tests for ActivityCreate model."""

    def test_minimal_activity_create(self):
        """Test creating activity with minimum required fields."""
        activity = ActivityCreate(
            type=ActivityType.NODE_CREATED,
            message="New observation created",
        )
        assert activity.type == ActivityType.NODE_CREATED
        assert activity.message == "New observation created"
        assert activity.node_id is None
        assert activity.status is None

    def test_full_activity_create(self):
        """Test creating activity with all fields."""
        suggestion = SuggestionData(
            from_node_id="node-1",
            from_node_type="Observation",
            from_node_label="First",
            to_node_id="node-2",
            to_node_type="Hypothesis",
            to_node_label="Second",
            relationship_type="SUPPORTS",
            confidence=0.8,
        )
        activity = ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="New relationship suggested",
            node_id="node-1",
            node_type="Observation",
            suggestion_data=suggestion,
            status=ActivityStatus.PENDING,
            created_by="system-llm",
            group_id="processing-123",
        )
        assert activity.type == ActivityType.RELATIONSHIP_SUGGESTED
        assert activity.suggestion_data is not None
        assert activity.status == ActivityStatus.PENDING
        assert activity.created_by == "system-llm"


class TestActivityUpdate:
    """Tests for ActivityUpdate model."""

    def test_update_status(self):
        """Test updating just the status."""
        update = ActivityUpdate(status=ActivityStatus.APPROVED)
        assert update.status == ActivityStatus.APPROVED
        assert update.message is None

    def test_update_with_feedback(self):
        """Test updating with user feedback."""
        update = ActivityUpdate(
            status=ActivityStatus.REJECTED,
            user_feedback="This relationship doesn't make sense",
        )
        assert update.status == ActivityStatus.REJECTED
        assert update.user_feedback == "This relationship doesn't make sense"


class TestActivityResponse:
    """Tests for ActivityResponse model."""

    def test_activity_response_from_dict(self):
        """Test creating response from dict."""
        response = ActivityResponse(
            id="activity-123",
            type=ActivityType.NODE_CREATED,
            message="Node created",
            created_at=datetime.now(UTC),
            node_id="node-1",
            node_type="Observation",
        )
        assert response.id == "activity-123"
        assert response.is_interactive is False
        assert response.has_navigation is True

    def test_interactive_activity(self):
        """Test is_interactive property for pending suggestion."""
        response = ActivityResponse(
            id="activity-123",
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Suggested relationship",
            created_at=datetime.now(UTC),
            status=ActivityStatus.PENDING,
        )
        assert response.is_interactive is True

    def test_non_interactive_approved_suggestion(self):
        """Test approved suggestion is no longer interactive."""
        response = ActivityResponse(
            id="activity-123",
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Suggested relationship",
            created_at=datetime.now(UTC),
            status=ActivityStatus.APPROVED,
        )
        assert response.is_interactive is False

    def test_has_navigation_with_node(self):
        """Test has_navigation with node_id."""
        response = ActivityResponse(
            id="activity-123",
            type=ActivityType.NODE_CREATED,
            message="Node created",
            created_at=datetime.now(UTC),
            node_id="node-1",
        )
        assert response.has_navigation is True

    def test_has_navigation_with_relationship(self):
        """Test has_navigation with relationship_id."""
        response = ActivityResponse(
            id="activity-123",
            type=ActivityType.RELATIONSHIP_CREATED,
            message="Relationship created",
            created_at=datetime.now(UTC),
            relationship_id="rel-1",
        )
        assert response.has_navigation is True

    def test_no_navigation(self):
        """Test has_navigation is False when no IDs."""
        response = ActivityResponse(
            id="activity-123",
            type=ActivityType.INFO,
            message="System info",
            created_at=datetime.now(UTC),
        )
        assert response.has_navigation is False


class TestActivityFilter:
    """Tests for ActivityFilter model."""

    def test_default_filter(self):
        """Test default filter values."""
        filter = ActivityFilter()
        assert filter.types is None
        assert filter.status is None
        assert filter.limit == 50
        assert filter.include_dismissed is False

    def test_filter_with_types(self):
        """Test filter with specific types."""
        filter = ActivityFilter(
            types=[ActivityType.NODE_CREATED, ActivityType.RELATIONSHIP_CREATED],
            limit=10,
        )
        assert len(filter.types) == 2
        assert ActivityType.NODE_CREATED in filter.types

    def test_filter_limit_validation(self):
        """Test filter limit max validation."""
        with pytest.raises(ValidationError):
            ActivityFilter(limit=500)  # Max is 200


class TestSuggestionThresholds:
    """Tests for SuggestionThresholds model."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = SuggestionThresholds()
        assert thresholds.auto_create_min == 0.8
        assert thresholds.suggest_min == 0.6

    def test_get_action_auto_create(self):
        """Test get_action returns auto_create for high confidence."""
        thresholds = SuggestionThresholds()
        assert thresholds.get_action(0.9) == "auto_create"
        assert thresholds.get_action(0.8) == "auto_create"

    def test_get_action_suggest(self):
        """Test get_action returns suggest for medium confidence."""
        thresholds = SuggestionThresholds()
        assert thresholds.get_action(0.7) == "suggest"
        assert thresholds.get_action(0.6) == "suggest"

    def test_get_action_discard(self):
        """Test get_action returns discard for low confidence."""
        thresholds = SuggestionThresholds()
        assert thresholds.get_action(0.5) == "discard"
        assert thresholds.get_action(0.1) == "discard"

    def test_default_thresholds_instance(self):
        """Test DEFAULT_THRESHOLDS is properly configured."""
        assert DEFAULT_THRESHOLDS.auto_create_min == 0.8
        assert DEFAULT_THRESHOLDS.suggest_min == 0.6

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        thresholds = SuggestionThresholds(
            auto_create_min=0.9,
            suggest_min=0.5,
        )
        assert thresholds.get_action(0.85) == "suggest"  # Not auto_create anymore
        assert thresholds.get_action(0.55) == "suggest"  # Still suggest
        assert thresholds.get_action(0.4) == "discard"

