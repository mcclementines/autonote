"""Markdown chunking service for hierarchical note splitting."""

import os
import re

import structlog
from opentelemetry import trace

from connectors.openai import OpenAIConnector, OpenAIModel

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


class MarkdownChunk:
    """Represents a single chunk of markdown content."""

    def __init__(
        self,
        content_md: str,
        heading_path: list[str],
        chunk_type: str,
        chunk_index: int,
    ):
        self.content_md = content_md
        self.heading_path = heading_path
        self.chunk_type = chunk_type
        self.chunk_index = chunk_index
        self.token_count = self._estimate_tokens(content_md)
        self.embedding = None

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count (rough: 4 chars per token)."""
        return len(text) // 4

    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB storage."""
        return {
            "content_md": self.content_md,
            "heading_path": self.heading_path,
            "chunk_type": self.chunk_type,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "embedding": self.embedding,
        }


class MarkdownChunker:
    """Split markdown notes into hierarchical chunks respecting structure."""

    def __init__(
        self,
        max_chunk_tokens: int | None = None,
        preserve_code_blocks: bool = True,
        min_chunk_tokens: int = 50,
    ):
        self.max_chunk_tokens = max_chunk_tokens or int(os.getenv("CHUNKING_MAX_TOKENS", "300"))
        self.preserve_code_blocks = preserve_code_blocks
        self.min_chunk_tokens = min_chunk_tokens

    @tracer.start_as_current_span("chunk_note")
    def chunk_note(self, note_content: str, note_title: str) -> list[MarkdownChunk]:
        """Split note into semantic chunks based on markdown structure.

        Args:
            note_content: The markdown content of the note
            note_title: The note title (used for context)

        Returns:
            List of MarkdownChunk objects
        """
        span = trace.get_current_span()
        span.set_attribute("note.title", note_title)
        span.set_attribute("note.content_length", len(note_content))

        logger.info(
            "chunking_note_start",
            title=note_title,
            content_length=len(note_content),
        )

        # Prepend title as H1 for context
        full_content = f"# {note_title}\n\n{note_content}"

        # Extract code blocks first (to avoid splitting them)
        content_with_placeholders, code_blocks = self._extract_code_blocks(full_content)

        # Split by headings
        sections = self._split_by_headings(content_with_placeholders)

        # Restore code blocks
        sections = self._restore_code_blocks(sections, code_blocks)

        # Process large sections
        chunks = []
        for section in sections:
            if section["token_count"] > self.max_chunk_tokens:
                # Split large section into smaller chunks
                sub_chunks = self._split_large_section(
                    section["content"],
                    section["heading_path"],
                    len(chunks),  # Starting index
                )
                chunks.extend(sub_chunks)
            else:
                # Section is already good size
                chunks.append(
                    MarkdownChunk(
                        content_md=section["content"],
                        heading_path=section["heading_path"],
                        chunk_type=section["chunk_type"],
                        chunk_index=len(chunks),
                    )
                )

        span.set_attribute("chunks.count", len(chunks))
        span.set_attribute(
            "chunks.avg_tokens",
            sum(c.token_count for c in chunks) // len(chunks) if chunks else 0,
        )

        logger.info(
            "chunking_note_complete",
            title=note_title,
            chunk_count=len(chunks),
        )

        return chunks

    def _extract_code_blocks(self, content: str) -> tuple[str, dict]:
        """Extract code blocks and replace with placeholders.

        Returns:
            Tuple of (content_with_placeholders, code_blocks_dict)
        """
        code_blocks = {}
        counter = [0]  # Use list for mutable counter in closure

        def replace_code_block(match):
            placeholder = f"__CODE_BLOCK_{counter[0]}__"
            code_blocks[placeholder] = match.group(0)
            counter[0] += 1
            return placeholder

        # Match code blocks: ```language\ncode\n```
        pattern = r"```[\s\S]*?```"
        content_with_placeholders = re.sub(pattern, replace_code_block, content)

        return content_with_placeholders, code_blocks

    def _restore_code_blocks(self, sections: list[dict], code_blocks: dict) -> list[dict]:
        """Restore code blocks from placeholders."""
        for section in sections:
            for placeholder, code_block in code_blocks.items():
                section["content"] = section["content"].replace(placeholder, code_block)
        return sections

    def _split_by_headings(self, content: str) -> list[dict]:
        """Split content by markdown headings (H1-H6).

        Returns:
            List of section dicts with heading_path, content, token_count
        """
        # Regex to match headings: # Title, ## Subtitle, etc.
        heading_pattern = r"^(#{1,6})\s+(.+?)$"

        lines = content.split("\n")
        sections = []
        current_section = {
            "heading_path": [],
            "content": "",
            "chunk_type": "section",
            "heading_level": 0,
        }
        heading_stack = []  # Track heading hierarchy

        for line in lines:
            match = re.match(heading_pattern, line, re.MULTILINE)

            if match:
                # Save previous section if it has content
                if current_section["content"].strip():
                    current_section["token_count"] = len(current_section["content"]) // 4
                    sections.append(current_section.copy())

                # Parse new heading
                level = len(match.group(1))
                heading_text = match.group(2).strip()

                # Update heading stack (maintain hierarchy)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading_text))

                # Create heading path from stack
                heading_path = [h[1] for h in heading_stack]

                # Start new section
                current_section = {
                    "heading_path": heading_path,
                    "content": line + "\n",
                    "chunk_type": "section",
                    "heading_level": level,
                }
            else:
                current_section["content"] += line + "\n"

        # Add final section
        if current_section["content"].strip():
            current_section["token_count"] = len(current_section["content"]) // 4
            sections.append(current_section)

        return sections

    def _split_large_section(
        self,
        content: str,
        heading_path: list[str],
        starting_index: int,
    ) -> list[MarkdownChunk]:
        """Split a large section into smaller chunks by paragraphs.

        Args:
            content: Section content
            heading_path: Breadcrumb path for this section
            starting_index: Chunk index to start from

        Returns:
            List of MarkdownChunk objects
        """
        # Split by paragraphs (double newline)
        paragraphs = re.split(r"\n\n+", content)

        chunks = []
        current_chunk = ""
        current_index = starting_index

        for para in paragraphs:
            if not para.strip():
                continue

            # Check if adding paragraph exceeds limit
            test_chunk = (current_chunk + "\n\n" + para).strip()
            test_tokens = len(test_chunk) // 4

            if test_tokens > self.max_chunk_tokens and current_chunk:
                # Save current chunk
                chunks.append(
                    MarkdownChunk(
                        content_md=current_chunk.strip(),
                        heading_path=heading_path,
                        chunk_type="paragraph",
                        chunk_index=current_index,
                    )
                )
                current_index += 1
                current_chunk = para
            else:
                # Add to current chunk
                current_chunk = test_chunk

        # Add final chunk
        if current_chunk.strip() and len(current_chunk) // 4 >= self.min_chunk_tokens:
            chunks.append(
                MarkdownChunk(
                    content_md=current_chunk.strip(),
                    heading_path=heading_path,
                    chunk_type="paragraph",
                    chunk_index=current_index,
                )
            )

        return chunks

    @tracer.start_as_current_span("generate_chunk_embeddings")
    async def generate_embeddings(
        self,
        chunks: list[MarkdownChunk],
        api_key: str | None = None,
    ) -> list[MarkdownChunk]:
        """Generate embeddings for all chunks.

        Args:
            chunks: List of MarkdownChunk objects
            api_key: OpenAI API key (uses env var if not provided)

        Returns:
            Same chunks list with embeddings populated
        """
        span = trace.get_current_span()
        span.set_attribute("chunks.count", len(chunks))

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("openai_api_key_missing_for_chunk_embeddings")
            return chunks

        try:
            async with OpenAIConnector(api_key=api_key) as connector:
                for i, chunk in enumerate(chunks):
                    embedding_response = await connector.embeddings(
                        input_text=chunk.content_md,
                        model=OpenAIModel.TEXT_EMBEDDING_3_SMALL,
                        dimensions=1536,
                    )
                    chunk.embedding = embedding_response.data[0].embedding

                    if (i + 1) % 10 == 0:
                        logger.debug("embeddings_generated", count=i + 1, total=len(chunks))

            span.set_attribute("embeddings.generated", True)
            logger.info("chunk_embeddings_complete", count=len(chunks))

        except Exception as e:
            logger.error("chunk_embedding_generation_failed", error=str(e))
            span.record_exception(e)
            span.set_attribute("embeddings.generated", False)

        return chunks
