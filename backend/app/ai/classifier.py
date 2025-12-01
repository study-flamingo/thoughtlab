"""Relationship classifier using LLM.

This module provides LLM-based relationship classification:
- Analyzes two pieces of content to determine relationship type
- Returns structured output with confidence scores
- Provides reasoning for the classification

Uses langchain-openai with structured output (Pydantic models).
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field
import logging

from langchain_openai import ChatOpenAI

from app.ai.config import AIConfig, get_ai_config

logger = logging.getLogger(__name__)


# Valid relationship types in ThoughtLab
RELATIONSHIP_TYPES = [
    "SUPPORTS",      # Source provides evidence for target
    "CONTRADICTS",   # Source conflicts with target
    "RELATES_TO",    # General topical connection
    "CITES",         # Source explicitly references target
    "DERIVED_FROM",  # Source was inspired by target
    "DISCUSSES",     # Source talks about same topic as target
    "PART_OF",       # Source is component of target
    "SIMILAR_TO",    # Content is semantically similar but distinct
]

RelationshipType = Literal[
    "SUPPORTS", "CONTRADICTS", "RELATES_TO", "CITES",
    "DERIVED_FROM", "DISCUSSES", "PART_OF", "SIMILAR_TO"
]


class RelationshipClassification(BaseModel):
    """Structured output for relationship classification.
    
    This model defines the expected response format from the LLM
    when classifying relationships between nodes.
    """
    
    relationship_type: RelationshipType = Field(
        description=(
            "The type of relationship: SUPPORTS, CONTRADICTS, RELATES_TO, "
            "CITES, DERIVED_FROM, DISCUSSES, PART_OF, or SIMILAR_TO"
        )
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score (0-1) for this classification"
    )
    reasoning: str = Field(
        description="Brief explanation for why this relationship exists"
    )
    is_valid: bool = Field(
        description="Whether a meaningful relationship exists between these nodes"
    )


class RelationshipClassifier:
    """Classifies relationships between nodes using LLM.
    
    Uses OpenAI's chat models with structured output to analyze
    content pairs and determine relationship types.
    """
    
    SYSTEM_PROMPT = """You are an expert at identifying relationships between 
pieces of information in a knowledge graph for research purposes.

Given two pieces of content, determine if there's a meaningful relationship.

## Relationship Types

- SUPPORTS: Source provides evidence for or validates target
- CONTRADICTS: Source conflicts with or challenges target
- RELATES_TO: General topical connection worth noting
- CITES: Source explicitly references target
- DERIVED_FROM: Source was inspired by or built upon target
- DISCUSSES: Source talks about the same topic as target
- PART_OF: Source is a component or subset of target
- SIMILAR_TO: Content is semantically similar but distinct

## Confidence Guidelines

Be conservative with confidence scores:
- 0.9+ : Very clear, explicit relationship with strong evidence
- 0.7-0.9: Strong implied relationship, high certainty
- 0.5-0.7: Possible relationship, moderate certainty
- <0.5: Weak or speculative - set is_valid to false

## Important

- Set is_valid=false if there's no meaningful research-relevant relationship
- Focus on relationships that would help a researcher understand connections
- Don't create relationships just because topics are vaguely related
- Be specific in your reasoning"""
    
    def __init__(self, config: Optional[AIConfig] = None):
        """Initialize relationship classifier.
        
        Args:
            config: AI configuration. If None, uses global config.
        """
        self.config = config or get_ai_config()
        self._llm = None
    
    @property
    def llm(self):
        """Lazy initialization of LLM with structured output."""
        if self._llm is None:
            if not self.config.is_configured:
                raise RuntimeError(
                    "OpenAI API key not configured. "
                    "Set THOUGHTLAB_OPENAI_API_KEY environment variable."
                )
            
            base_llm = ChatOpenAI(
                model=self.config.llm_model,
                openai_api_key=self.config.openai_api_key,
                temperature=self.config.llm_temperature,
            )
            self._llm = base_llm.with_structured_output(RelationshipClassification)
            logger.info(f"Initialized relationship classifier: {self.config.llm_model}")
        
        return self._llm
    
    @property
    def is_ready(self) -> bool:
        """Check if classifier is ready."""
        return self.config.is_configured
    
    async def classify(
        self,
        source_content: str,
        source_type: str,
        target_content: str,
        target_type: str,
        max_content_length: int = 1000,
    ) -> Optional[RelationshipClassification]:
        """Classify the relationship between two pieces of content.
        
        Args:
            source_content: Content of the source node.
            source_type: Type of the source node (e.g., "Observation").
            target_content: Content of the target node.
            target_type: Type of the target node.
            max_content_length: Maximum characters to include from each content.
                               Truncates to avoid token limits.
        
        Returns:
            RelationshipClassification with type, confidence, and reasoning.
            Returns None if classification fails.
        """
        if not self.is_ready:
            logger.warning("Relationship classifier not ready (AI not configured)")
            return None
        
        # Truncate content to avoid token limits
        source_preview = source_content[:max_content_length]
        if len(source_content) > max_content_length:
            source_preview += "..."
            
        target_preview = target_content[:max_content_length]
        if len(target_content) > max_content_length:
            target_preview += "..."
        
        prompt = f"""Analyze if there's a meaningful relationship between these two items.

## Source ({source_type})
{source_preview}

## Target ({target_type})
{target_preview}

Determine the relationship type, confidence, and provide reasoning."""
        
        try:
            result = await self.llm.ainvoke([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            
            logger.debug(
                f"Classified relationship: {result.relationship_type} "
                f"(confidence={result.confidence}, valid={result.is_valid})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Relationship classification failed: {e}")
            return None
    
    async def classify_batch(
        self,
        pairs: list[tuple[str, str, str, str]],
        max_content_length: int = 1000,
    ) -> list[Optional[RelationshipClassification]]:
        """Classify multiple content pairs.
        
        Note: Currently processes sequentially. Could be parallelized
        for better performance with rate limiting.
        
        Args:
            pairs: List of (source_content, source_type, target_content, target_type).
            max_content_length: Maximum characters per content piece.
            
        Returns:
            List of classifications (None for failures).
        """
        results = []
        for source_content, source_type, target_content, target_type in pairs:
            result = await self.classify(
                source_content=source_content,
                source_type=source_type,
                target_content=target_content,
                target_type=target_type,
                max_content_length=max_content_length,
            )
            results.append(result)
        
        return results


# Global instance (lazy initialization)
_classifier: Optional[RelationshipClassifier] = None


def get_relationship_classifier() -> RelationshipClassifier:
    """Get the global relationship classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = RelationshipClassifier()
    return _classifier

