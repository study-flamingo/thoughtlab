"""Unit tests for ToolService.

These tests use mocked dependencies to test service logic without requiring
Neo4j or OpenAI API calls.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.tools import ToolService
from app.services.tools.base import ToolServiceBase
from app.models.tool_models import (
    RelatedNodeResult,
    FindRelatedNodesResponse,
    SummarizeNodeResponse,
    SummarizeNodeWithContextResponse,
    NodeContextSummary,
    RecalculateConfidenceResponse,
    ConfidenceFactor,
    ReclassifyNodeResponse,
    SummarizeRelationshipResponse,
    NodeInfo,
    RecalculateEdgeConfidenceResponse,
    ReclassifyRelationshipResponse,
    MergeNodesResponse,
    SearchWebEvidenceResponse,
)


# ============================================================================
# Response Model Tests
# ============================================================================

class TestResponseModels:
    """Test Pydantic response models."""

    def test_related_node_result_model(self):
        """Test RelatedNodeResult model validation."""
        result = RelatedNodeResult(
            id="node-123",
            type="Observation",
            content="Test content",
            similarity_score=0.85,
            suggested_relationship="SUPPORTS",
            reasoning="High semantic similarity"
        )
        assert result.id == "node-123"
        assert result.similarity_score == 0.85

    def test_find_related_nodes_response(self):
        """Test FindRelatedNodesResponse model."""
        response = FindRelatedNodesResponse(
            success=True,
            node_id="obs-123",
            related_nodes=[],
            links_created=0,
            message="Found 0 related nodes"
        )
        assert response.success is True
        assert response.related_nodes == []

    def test_summarize_node_response(self):
        """Test SummarizeNodeResponse model."""
        response = SummarizeNodeResponse(
            success=True,
            node_id="obs-123",
            summary="A summary of the node",
            key_points=["Point 1", "Point 2"],
            word_count=5
        )
        assert response.summary == "A summary of the node"
        assert len(response.key_points) == 2

    def test_summarize_node_with_context_response(self):
        """Test SummarizeNodeWithContextResponse model."""
        context = NodeContextSummary(
            supports=["Supporting node 1"],
            contradicts=[],
            related=["Related node 1", "Related node 2"]
        )
        response = SummarizeNodeWithContextResponse(
            success=True,
            node_id="hyp-123",
            summary="Hypothesis summary",
            context=context,
            synthesis="Overall synthesis",
            relationship_count=3
        )
        assert len(response.context.supports) == 1
        assert len(response.context.related) == 2

    def test_recalculate_confidence_response(self):
        """Test RecalculateConfidenceResponse model."""
        response = RecalculateConfidenceResponse(
            success=True,
            node_id="obs-123",
            old_confidence=0.7,
            new_confidence=0.85,
            reasoning="Increased due to supporting evidence",
            factors=[
                ConfidenceFactor(factor="Supporting evidence", impact="+0.1"),
                ConfidenceFactor(factor="Source credibility", impact="+0.05")
            ]
        )
        assert response.old_confidence == 0.7
        assert response.new_confidence == 0.85
        assert len(response.factors) == 2

    def test_reclassify_node_response(self):
        """Test ReclassifyNodeResponse model."""
        response = ReclassifyNodeResponse(
            success=True,
            node_id="obs-123",
            old_type="Observation",
            new_type="Hypothesis",
            properties_preserved=["text", "confidence", "created_at"],
            relationships_preserved=5,
            message="Successfully reclassified"
        )
        assert response.old_type == "Observation"
        assert response.new_type == "Hypothesis"
        assert len(response.properties_preserved) == 3

    def test_summarize_relationship_response(self):
        """Test SummarizeRelationshipResponse model."""
        response = SummarizeRelationshipResponse(
            success=True,
            edge_id="rel-123",
            from_node=NodeInfo(id="obs-1", type="Observation", content="Content 1"),
            to_node=NodeInfo(id="hyp-1", type="Hypothesis", content="Content 2"),
            relationship_type="SUPPORTS",
            summary="The observation supports the hypothesis",
            evidence=["Evidence 1", "Evidence 2"],
            strength_assessment="strong"
        )
        assert response.relationship_type == "SUPPORTS"
        assert response.strength_assessment == "strong"

    def test_recalculate_edge_confidence_response(self):
        """Test RecalculateEdgeConfidenceResponse model."""
        response = RecalculateEdgeConfidenceResponse(
            success=True,
            edge_id="rel-123",
            old_confidence=0.6,
            new_confidence=0.75,
            reasoning="Strong semantic connection",
            factors=[ConfidenceFactor(factor="Content alignment", impact="good")]
        )
        assert response.old_confidence == 0.6
        assert response.new_confidence == 0.75

    def test_reclassify_relationship_response(self):
        """Test ReclassifyRelationshipResponse model."""
        response = ReclassifyRelationshipResponse(
            success=True,
            edge_id="rel-123",
            old_type="RELATES_TO",
            new_type="SUPPORTS",
            suggested_by_ai=True,
            reasoning="AI suggested SUPPORTS based on content analysis",
            notes_preserved=True
        )
        assert response.suggested_by_ai is True
        assert response.new_type == "SUPPORTS"

    def test_merge_nodes_response(self):
        """Test MergeNodesResponse model."""
        response = MergeNodesResponse(
            success=True,
            primary_node_id="obs-1",
            secondary_node_id="obs-2",
            merged_properties=["text", "notes"],
            relationships_transferred=3,
            message="Successfully merged nodes"
        )
        assert len(response.merged_properties) == 2
        assert response.relationships_transferred == 3

    def test_search_web_evidence_response(self):
        """Test SearchWebEvidenceResponse model."""
        response = SearchWebEvidenceResponse(
            success=False,
            node_id="hyp-123",
            query_used="evidence for quantum entanglement",
            results=[],
            sources_created=0,
            message="Web search not configured"
        )
        assert response.success is False
        assert response.results == []


# ============================================================================
# Helper Method Tests
# ============================================================================

class TestExtractNodeContent:
    """Test extract_node_content helper method."""

    def test_extract_text_field(self):
        """Test extraction from text field."""
        base = ToolServiceBase()
        node = {"text": "This is the text content", "type": "Observation"}
        content = base.extract_node_content(node)
        assert content == "This is the text content"

    def test_extract_description_field(self):
        """Test extraction from description field."""
        base = ToolServiceBase()
        node = {"description": "This is the description", "type": "Entity"}
        content = base.extract_node_content(node)
        assert content == "This is the description"

    def test_extract_title_field(self):
        """Test extraction from title field."""
        base = ToolServiceBase()
        node = {"title": "Source Title", "type": "Source"}
        content = base.extract_node_content(node)
        assert content == "Source Title"

    def test_extract_name_field(self):
        """Test extraction from name field."""
        base = ToolServiceBase()
        node = {"name": "Hypothesis Name", "type": "Hypothesis"}
        content = base.extract_node_content(node)
        assert content == "Hypothesis Name"

    def test_extract_priority_order(self):
        """Test that text field has priority over others."""
        base = ToolServiceBase()
        node = {
            "text": "Primary text",
            "description": "Secondary description",
            "title": "Tertiary title",
            "type": "Observation"
        }
        content = base.extract_node_content(node)
        assert content == "Primary text"

    def test_extract_empty_node(self):
        """Test extraction from node with no content fields."""
        base = ToolServiceBase()
        node = {"type": "Observation", "id": "obs-123"}
        content = base.extract_node_content(node)
        assert content == ""

    def test_extract_none_values(self):
        """Test extraction skips None values."""
        base = ToolServiceBase()
        node = {"text": None, "description": "Description", "type": "Entity"}
        content = base.extract_node_content(node)
        assert content == "Description"


# ============================================================================
# Service Method Tests with Mocks
# ============================================================================

class TestFindRelatedNodes:
    """Test find_related_nodes method."""

    @pytest.mark.asyncio
    async def test_find_related_nodes_not_found(self):
        """Test find_related_nodes returns error for non-existent node."""
        service = ToolService()

        with patch.object(service._node_analysis, 'get_node_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = await service.find_related_nodes("nonexistent-id")

            assert response.success is False
            assert response.error == "Node not found"
            assert response.related_nodes == []

    @pytest.mark.asyncio
    async def test_find_related_nodes_no_content(self):
        """Test find_related_nodes handles node with no content."""
        service = ToolService()

        with patch.object(service._node_analysis, 'get_node_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"id": "obs-123", "type": "Observation"}  # No text

            response = await service.find_related_nodes("obs-123")

            assert response.success is False
            assert "no content" in response.message.lower()


class TestSummarizeNode:
    """Test summarize_node method."""

    @pytest.mark.asyncio
    async def test_summarize_node_not_found(self):
        """Test summarize_node returns error for non-existent node."""
        service = ToolService()

        with patch.object(service._node_analysis, 'get_node_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = await service.summarize_node("nonexistent-id")

            assert response.success is False
            assert response.error == "Node not found"
            assert response.summary == ""

    @pytest.mark.asyncio
    async def test_summarize_node_no_content(self):
        """Test summarize_node handles node with no content."""
        service = ToolService()

        with patch.object(service._node_analysis, 'get_node_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"id": "obs-123", "type": "Observation"}  # No text

            response = await service.summarize_node("obs-123")

            assert response.success is False
            assert "no content" in response.error.lower()


class TestReclassifyNode:
    """Test reclassify_node method."""

    @pytest.mark.asyncio
    async def test_reclassify_node_not_found(self):
        """Test reclassify_node returns error for non-existent node."""
        service = ToolService()

        with patch.object(service._node_modification, 'get_node_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = await service.reclassify_node("nonexistent-id", "Hypothesis")

            assert response.success is False
            assert response.error == "Node not found"

    @pytest.mark.asyncio
    async def test_reclassify_node_invalid_type(self):
        """Test reclassify_node rejects invalid node type."""
        service = ToolService()

        with patch.object(service._node_modification, 'get_node_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"id": "obs-123", "type": "Observation", "text": "Test"}

            response = await service.reclassify_node("obs-123", "InvalidType")

            assert response.success is False
            assert "invalid" in response.message.lower()

    @pytest.mark.asyncio
    async def test_reclassify_node_same_type(self):
        """Test reclassify_node handles same type gracefully."""
        service = ToolService()

        with patch.object(service._node_modification, 'get_node_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"id": "obs-123", "type": "Observation", "text": "Test"}

            response = await service.reclassify_node("obs-123", "Observation")

            assert response.success is True
            assert response.old_type == "Observation"
            assert response.new_type == "Observation"
            assert "already" in response.message.lower()


class TestMergeNodes:
    """Test merge_nodes method."""

    @pytest.mark.asyncio
    async def test_merge_nodes_primary_not_found(self):
        """Test merge_nodes returns error when primary node not found."""
        service = ToolService()

        with patch.object(service._node_modification, 'get_node_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = await service.merge_nodes("nonexistent-1", "nonexistent-2")

            assert response.success is False
            assert "primary node not found" in response.message.lower()

    @pytest.mark.asyncio
    async def test_merge_nodes_secondary_not_found(self):
        """Test merge_nodes returns error when secondary node not found."""
        service = ToolService()

        async def mock_get_node(node_id):
            if node_id == "obs-1":
                return {"id": "obs-1", "type": "Observation", "text": "Primary"}
            return None

        with patch.object(service._node_modification, 'get_node_by_id', side_effect=mock_get_node):
            response = await service.merge_nodes("obs-1", "nonexistent-2")

            assert response.success is False
            assert "secondary node not found" in response.message.lower()

    @pytest.mark.asyncio
    async def test_merge_nodes_type_mismatch(self):
        """Test merge_nodes rejects nodes of different types."""
        service = ToolService()

        async def mock_get_node(node_id):
            if node_id == "obs-1":
                return {"id": "obs-1", "type": "Observation", "text": "Observation text"}
            elif node_id == "hyp-1":
                return {"id": "hyp-1", "type": "Hypothesis", "name": "Hypothesis name"}
            return None

        with patch.object(service._node_modification, 'get_node_by_id', side_effect=mock_get_node):
            response = await service.merge_nodes("obs-1", "hyp-1")

            assert response.success is False
            assert "different types" in response.message.lower()
            assert response.error == "Node type mismatch"


class TestSummarizeRelationship:
    """Test summarize_relationship method."""

    @pytest.mark.asyncio
    async def test_summarize_relationship_not_found(self):
        """Test summarize_relationship returns error for non-existent relationship."""
        service = ToolService()

        with patch.object(service._relationship_analysis, 'get_relationship_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = await service.summarize_relationship("nonexistent-rel")

            assert response.success is False
            assert response.error == "Relationship not found"


class TestRecalculateEdgeConfidence:
    """Test recalculate_edge_confidence method."""

    @pytest.mark.asyncio
    async def test_recalculate_edge_confidence_not_found(self):
        """Test recalculate_edge_confidence returns error for non-existent relationship."""
        service = ToolService()

        with patch.object(service._relationship_analysis, 'get_relationship_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = await service.recalculate_edge_confidence("nonexistent-rel")

            assert response.success is False
            assert response.error == "Relationship not found"


class TestReclassifyRelationship:
    """Test reclassify_relationship method."""

    @pytest.mark.asyncio
    async def test_reclassify_relationship_not_found(self):
        """Test reclassify_relationship returns error for non-existent relationship."""
        service = ToolService()

        with patch.object(service._relationship_analysis, 'get_relationship_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = await service.reclassify_relationship("nonexistent-rel")

            assert response.success is False
            assert response.error == "Relationship not found"


class TestSearchWebEvidence:
    """Test search_web_evidence method."""

    @pytest.mark.asyncio
    async def test_search_web_evidence_no_api_key(self):
        """Test search_web_evidence returns error when TAVILY_API_KEY not set."""
        service = ToolService()

        with patch.dict('os.environ', {}, clear=True):
            with patch.object(service._node_modification, 'get_node_by_id', new_callable=AsyncMock) as mock_get:
                mock_get.return_value = {"id": "hyp-123", "type": "Hypothesis", "text": "Test hypothesis"}

                response = await service.search_web_evidence("hyp-123")

                assert response.success is False
                assert "not configured" in response.message.lower()

    @pytest.mark.asyncio
    async def test_search_web_evidence_node_not_found(self):
        """Test search_web_evidence returns error when node not found."""
        service = ToolService()

        with patch.dict('os.environ', {'TAVILY_API_KEY': 'test-key'}):
            with patch.object(service._node_modification, 'get_node_by_id', new_callable=AsyncMock) as mock_get:
                mock_get.return_value = None

                response = await service.search_web_evidence("nonexistent-id")

                assert response.success is False
                assert "not found" in response.error.lower()
