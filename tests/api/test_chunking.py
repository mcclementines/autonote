"""Unit tests for markdown chunking service."""

import pytest

from api.services.chunking import MarkdownChunk, MarkdownChunker


class TestMarkdownChunk:
    """Test suite for MarkdownChunk class."""

    def test_chunk_initialization(self):
        """Test basic chunk initialization."""
        chunk = MarkdownChunk(
            content_md="Test content",
            heading_path=["Section 1"],
            chunk_type="section",
            chunk_index=0,
        )

        assert chunk.content_md == "Test content"
        assert chunk.heading_path == ["Section 1"]
        assert chunk.chunk_type == "section"
        assert chunk.chunk_index == 0
        assert chunk.embedding is None

    def test_token_count_estimation(self):
        """Test token count estimation (4 chars per token)."""
        # 40 characters = ~10 tokens
        chunk = MarkdownChunk(
            content_md="1234567890" * 4,  # 40 chars
            heading_path=[],
            chunk_type="section",
            chunk_index=0,
        )

        assert chunk.token_count == 10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        chunk = MarkdownChunk(
            content_md="Test",
            heading_path=["H1"],
            chunk_type="section",
            chunk_index=0,
        )
        chunk.embedding = [0.1, 0.2, 0.3]

        result = chunk.to_dict()

        assert result["content_md"] == "Test"
        assert result["heading_path"] == ["H1"]
        assert result["chunk_type"] == "section"
        assert result["chunk_index"] == 0
        assert result["token_count"] == 1
        assert result["embedding"] == [0.1, 0.2, 0.3]


class TestMarkdownChunker:
    """Test suite for MarkdownChunker class."""

    def test_chunker_initialization(self):
        """Test chunker initialization with default values."""
        chunker = MarkdownChunker()

        assert chunker.max_chunk_tokens == 300
        assert chunker.preserve_code_blocks is True
        assert chunker.min_chunk_tokens == 50

    def test_custom_chunk_size(self):
        """Test chunker with custom chunk size."""
        chunker = MarkdownChunker(max_chunk_tokens=500)

        assert chunker.max_chunk_tokens == 500

    def test_simple_note_chunking(self):
        """Test chunking a simple note with one section."""
        chunker = MarkdownChunker(max_chunk_tokens=500)

        content = "This is a simple note without headings."
        chunks = chunker.chunk_note(content, "Simple Note")

        assert len(chunks) >= 1
        assert isinstance(chunks[0], MarkdownChunk)
        # Title should be prepended as H1
        assert "Simple Note" in chunks[0].content_md

    def test_heading_based_splitting(self):
        """Test splitting by markdown headings."""
        chunker = MarkdownChunker(max_chunk_tokens=500)

        content = """
## Introduction
This is the introduction section.

## Methods
This is the methods section.

## Conclusion
This is the conclusion section.
"""

        chunks = chunker.chunk_note(content, "Test Note")

        # Should create multiple chunks for different sections
        assert len(chunks) >= 3

        # Check heading paths
        heading_paths = [chunk.heading_path for chunk in chunks]
        assert any("Introduction" in path for path in heading_paths)
        assert any("Methods" in path for path in heading_paths)
        assert any("Conclusion" in path for path in heading_paths)

    def test_nested_headings(self):
        """Test nested heading hierarchy preservation."""
        chunker = MarkdownChunker(max_chunk_tokens=500)

        content = """
# Title

## Section 1

### Subsection 1.1

Content for subsection 1.1.

### Subsection 1.2

Content for subsection 1.2.

## Section 2

Final content.
"""

        chunks = chunker.chunk_note(content, "Nested Test")

        # Find subsection chunks
        subsection_chunks = [c for c in chunks if len(c.heading_path) >= 3]

        assert len(subsection_chunks) >= 2

        # Verify breadcrumb paths
        paths_str = [" > ".join(c.heading_path) for c in subsection_chunks]
        assert any("Section 1" in path and "Subsection 1.1" in path for path in paths_str)

    def test_code_block_preservation(self):
        """Test that code blocks stay together and aren't split."""
        chunker = MarkdownChunker(max_chunk_tokens=100)

        content = """
## Code Example

Here is some code:

```python
def hello():
    print("world")
    return True
```

This is after the code.
"""

        chunks = chunker.chunk_note(content, "Code Test")

        # Find chunks containing code
        code_chunks = [c for c in chunks if "```python" in c.content_md]

        assert len(code_chunks) >= 1

        # Verify entire code block is in one chunk
        code_chunk = code_chunks[0]
        assert "def hello():" in code_chunk.content_md
        assert "return True" in code_chunk.content_md
        assert "```" in code_chunk.content_md

    def test_large_section_splitting(self):
        """Test splitting of sections that exceed max tokens."""
        chunker = MarkdownChunker(max_chunk_tokens=50)

        # Create large content (will exceed 50 tokens)
        large_text = " ".join(["This is a test sentence."] * 50)
        content = f"""
## Large Section

{large_text}
"""

        chunks = chunker.chunk_note(content, "Large Test")

        # Should split into multiple chunks
        assert len(chunks) > 1

        # All chunks should have same heading path
        heading_paths = [tuple(c.heading_path) for c in chunks if c.heading_path]
        if heading_paths:
            # Check that chunks from same section have same path
            large_section_chunks = [
                c for c in chunks if c.heading_path and "Large Section" in c.heading_path
            ]
            if len(large_section_chunks) > 1:
                first_path = large_section_chunks[0].heading_path
                assert all(c.heading_path == first_path for c in large_section_chunks[1:]), (
                    "Chunks from same section should have same heading path"
                )

    def test_chunk_indices_sequential(self):
        """Test that chunk indices are sequential."""
        chunker = MarkdownChunker()

        content = """
## Section 1
Content 1

## Section 2
Content 2

## Section 3
Content 3
"""

        chunks = chunker.chunk_note(content, "Index Test")

        # Verify indices are sequential
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_content(self):
        """Test handling of empty content."""
        chunker = MarkdownChunker()

        chunks = chunker.chunk_note("", "Empty Note")

        # Should create at least one chunk with the title
        assert len(chunks) >= 1

    def test_content_with_only_code_block(self):
        """Test note that is only a code block."""
        chunker = MarkdownChunker()

        content = """```python
def main():
    pass
```"""

        chunks = chunker.chunk_note(content, "Code Only")

        assert len(chunks) >= 1
        assert "```python" in chunks[0].content_md or any(
            "```python" in c.content_md for c in chunks
        )

    def test_mixed_heading_levels(self):
        """Test content with mixed heading levels."""
        chunker = MarkdownChunker()

        content = """
# H1 Title

Some content.

### H3 Subsection

Skipped H2.

## H2 Section

Back to H2.
"""

        chunks = chunker.chunk_note(content, "Mixed Headings")

        # Should handle gracefully without errors
        assert len(chunks) >= 1
        assert all(isinstance(c.heading_path, list) for c in chunks)

    def test_special_markdown_characters(self):
        """Test handling of special markdown characters."""
        chunker = MarkdownChunker()

        content = """
## Section with **bold** and *italic*

Content with `inline code` and [links](http://example.com).

- List item 1
- List item 2

> Quote block
"""

        chunks = chunker.chunk_note(content, "Special Chars")

        # Should preserve markdown formatting
        combined_content = "".join(c.content_md for c in chunks)
        assert "**bold**" in combined_content
        assert "`inline code`" in combined_content


@pytest.mark.asyncio
class TestChunkEmbeddings:
    """Test suite for embedding generation."""

    async def test_embedding_generation_without_api_key(self):
        """Test that embedding generation handles missing API key gracefully."""
        chunker = MarkdownChunker()

        chunks = [
            MarkdownChunk(
                content_md="Test content",
                heading_path=["Test"],
                chunk_type="section",
                chunk_index=0,
            )
        ]

        # Should not fail, just skip embedding generation
        result = await chunker.generate_embeddings(chunks, api_key=None)

        assert len(result) == 1
        assert result[0].embedding is None

    # Note: Actual API tests would require mocking OpenAI API
    # or having a valid API key. Skipping for now.


class TestCodeBlockExtraction:
    """Test suite for code block extraction and restoration."""

    def test_code_block_extraction(self):
        """Test extracting code blocks as placeholders."""
        chunker = MarkdownChunker()

        content = """
Text before code.

```python
def test():
    pass
```

Text after code.

```javascript
console.log("test");
```

Final text.
"""

        content_with_placeholders, code_blocks = chunker._extract_code_blocks(content)

        # Should have 2 code blocks
        assert len(code_blocks) == 2

        # Placeholders should be in content
        assert "__CODE_BLOCK_0__" in content_with_placeholders
        assert "__CODE_BLOCK_1__" in content_with_placeholders

        # Code blocks should be preserved
        assert "def test():" in str(code_blocks.values())
        assert 'console.log("test");' in str(code_blocks.values())

    def test_code_block_restoration(self):
        """Test restoring code blocks from placeholders."""
        chunker = MarkdownChunker()

        sections = [
            {
                "content": "Text __CODE_BLOCK_0__ more text",
                "heading_path": [],
                "chunk_type": "section",
            }
        ]

        code_blocks = {"__CODE_BLOCK_0__": "```python\ncode\n```"}

        result = chunker._restore_code_blocks(sections, code_blocks)

        assert "```python\ncode\n```" in result[0]["content"]
        assert "__CODE_BLOCK_0__" not in result[0]["content"]


class TestHeadingSplitting:
    """Test suite for heading-based splitting logic."""

    def test_split_by_single_heading(self):
        """Test splitting content with one heading."""
        chunker = MarkdownChunker()

        content = """
## Heading
Content under heading.
"""

        sections = chunker._split_by_headings(content)

        assert len(sections) >= 1
        assert any("Heading" in s["heading_path"] for s in sections)

    def test_split_preserves_content(self):
        """Test that splitting preserves all content."""
        chunker = MarkdownChunker()

        original_content = """
## Section 1
Content 1

## Section 2
Content 2
"""

        sections = chunker._split_by_headings(original_content)

        # Combine all section content
        combined = "".join(s["content"] for s in sections)

        # Should contain all original text
        assert "Section 1" in combined
        assert "Content 1" in combined
        assert "Section 2" in combined
        assert "Content 2" in combined


class TestLargeSectionSplitting:
    """Test suite for splitting large sections."""

    def test_split_large_section_by_paragraphs(self):
        """Test splitting large section into paragraph chunks."""
        chunker = MarkdownChunker(max_chunk_tokens=50)

        # Create content with multiple paragraphs
        content = "\n\n".join([f"Paragraph {i}. " + "Text. " * 20 for i in range(5)])

        chunks = chunker._split_large_section(
            content=content, heading_path=["Test"], starting_index=0
        )

        # Should create multiple chunks
        assert len(chunks) > 1

        # All chunks should have same heading path
        assert all(c.heading_path == ["Test"] for c in chunks)

        # Indices should be sequential
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_min_chunk_size_respected(self):
        """Test that very small chunks are filtered out."""
        chunker = MarkdownChunker(min_chunk_tokens=50)

        # Content that would create tiny chunks
        content = "A.\n\nB.\n\nC."

        chunks = chunker._split_large_section(
            content=content, heading_path=["Test"], starting_index=0
        )

        # Small chunks should be filtered or combined
        # This is somewhat implementation-dependent
        assert all(c.token_count >= chunker.min_chunk_tokens for c in chunks)
