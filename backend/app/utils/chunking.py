"""Text chunking utilities for splitting long content into embeddable chunks.

This module provides chunking strategies for breaking down long-form content
(like articles, papers, or documents) into smaller pieces suitable for:
- Vector embedding (most models have token limits)
- Granular retrieval (find specific parts of a document)
- LLM context windows (process manageable pieces)

Current Implementation:
- RecursiveCharacterSplitter: Splits text by trying multiple separators
  (paragraphs, sentences, words) to find natural break points.

Future Considerations:
- Semantic chunking (split by meaning using embeddings)
- Document-aware chunking (respect markdown headers, code blocks)
"""

from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class Chunk:
    """A chunk of text with position tracking."""
    content: str
    chunk_index: int
    start_char: int  # Position in original text
    end_char: int    # Position in original text
    metadata: Optional[dict] = None
    
    @property
    def char_count(self) -> int:
        return len(self.content)
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Chunk({self.chunk_index}, chars={self.char_count}, '{preview}')"


class RecursiveCharacterSplitter:
    """Recursively split text using a hierarchy of separators.
    
    Tries to split on larger semantic boundaries first (paragraphs),
    then falls back to smaller boundaries (sentences, words, characters)
    when chunks are still too large.
    
    Based on LangChain's RecursiveCharacterTextSplitter concept.
    
    Example:
        splitter = RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text("Long article content here...")
    """
    
    # Default separators in order of preference (try larger units first)
    DEFAULT_SEPARATORS = [
        "\n\n",      # Double newline (paragraphs)
        "\n",        # Single newline
        ". ",        # Sentences (with space to avoid decimals)
        "? ",        # Questions
        "! ",        # Exclamations
        "; ",        # Semicolons
        ", ",        # Commas
        " ",         # Words
        "",          # Characters (last resort)
    ]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        length_function: Optional[callable] = None,
        keep_separator: bool = True,
        strip_whitespace: bool = True,
    ):
        """Initialize the splitter.
        
        Args:
            chunk_size: Maximum size of each chunk (in characters by default)
            chunk_overlap: Overlap between chunks for context continuity
            separators: List of separators to try, in order of preference
            length_function: Function to measure text length (default: len)
            keep_separator: Whether to keep separators in the chunks
            strip_whitespace: Whether to strip whitespace from chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.length_function = length_function or len
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace
    
    def split_text(self, text: str) -> List[Chunk]:
        """Split text into chunks.
        
        Args:
            text: The text to split
            
        Returns:
            List of Chunk objects with position tracking
        """
        if not text:
            return []
        
        # Get raw splits first
        splits = self._split_text_recursive(text, self.separators)
        
        # Merge small splits and create chunks with overlap
        chunks = self._merge_splits(splits, text)
        
        return chunks
    
    def _split_text_recursive(
        self,
        text: str,
        separators: List[str],
    ) -> List[str]:
        """Recursively split text using separators."""
        final_chunks: List[str] = []
        
        # Find the best separator for this text
        separator = separators[-1]  # Default to last (smallest unit)
        new_separators: List[str] = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break
        
        # Split on this separator
        if separator:
            splits = text.split(separator)
        else:
            # Character-by-character split
            splits = list(text)
        
        # Process each split
        good_splits: List[str] = []
        separator_to_use = separator if self.keep_separator else ""
        
        for split in splits:
            if self.length_function(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # This split is too big, need to recurse
                if good_splits:
                    merged = self._merge_small_splits(good_splits, separator_to_use)
                    final_chunks.extend(merged)
                    good_splits = []
                
                if new_separators:
                    # Recurse with smaller separators
                    sub_splits = self._split_text_recursive(split, new_separators)
                    final_chunks.extend(sub_splits)
                else:
                    # Can't split further, add as-is
                    final_chunks.append(split)
        
        # Don't forget remaining good splits
        if good_splits:
            merged = self._merge_small_splits(good_splits, separator_to_use)
            final_chunks.extend(merged)
        
        return final_chunks
    
    def _merge_small_splits(
        self,
        splits: List[str],
        separator: str,
    ) -> List[str]:
        """Merge small splits together up to chunk_size."""
        merged: List[str] = []
        current: List[str] = []
        current_len = 0
        
        for split in splits:
            split_len = self.length_function(split)
            sep_len = self.length_function(separator) if current else 0
            
            if current_len + split_len + sep_len > self.chunk_size:
                # Current chunk is full, start new one
                if current:
                    merged.append(separator.join(current))
                current = [split]
                current_len = split_len
            else:
                current.append(split)
                current_len += split_len + sep_len
        
        # Don't forget the last chunk
        if current:
            merged.append(separator.join(current))
        
        return merged
    
    def _merge_splits(
        self,
        splits: List[str],
        original_text: str,
    ) -> List[Chunk]:
        """Merge splits into final chunks with overlap and position tracking."""
        if not splits:
            return []
        
        chunks: List[Chunk] = []
        current_pos = 0
        chunk_index = 0
        
        i = 0
        while i < len(splits):
            # Build current chunk
            chunk_text = splits[i]
            
            # Apply strip if needed
            if self.strip_whitespace:
                chunk_text = chunk_text.strip()
            
            if not chunk_text:
                i += 1
                continue
            
            # Find position in original text
            start_char = original_text.find(chunk_text, current_pos)
            if start_char == -1:
                # Fallback: use current position
                start_char = current_pos
            
            end_char = start_char + len(chunk_text)
            
            chunks.append(Chunk(
                content=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
            ))
            
            # Update position (account for overlap)
            if self.chunk_overlap > 0 and end_char > self.chunk_overlap:
                current_pos = end_char - self.chunk_overlap
            else:
                current_pos = end_char
            
            chunk_index += 1
            i += 1
        
        return chunks


# Convenience function for common use case
def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """Convenience function to chunk text with default settings.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Chunk objects
    """
    splitter = RecursiveCharacterSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


# Threshold for when to chunk content (don't chunk small content)
MIN_CONTENT_LENGTH_FOR_CHUNKING = 500  # Characters


def should_chunk(text: str, threshold: int = MIN_CONTENT_LENGTH_FOR_CHUNKING) -> bool:
    """Determine if text should be chunked.
    
    Small content doesn't benefit from chunking and can be embedded directly.
    """
    return len(text) > threshold

