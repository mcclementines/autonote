"""System prompts for RAG and chat."""

from pathlib import Path


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        filename: Name of the prompt file (e.g., 'rag_system_prompt.txt')

    Returns:
        Prompt template as string
    """
    prompt_path = Path(__file__).parent / filename
    return prompt_path.read_text().strip()


def get_rag_system_prompt(context: str) -> str:
    """Get the RAG system prompt with context filled in.

    Args:
        context: Formatted context string with retrieved notes

    Returns:
        Complete system prompt
    """
    template = load_prompt("rag_system_prompt.txt")
    return template.format(context=context)


def get_default_system_prompt() -> str:
    """Get the default system prompt (no RAG context).

    Returns:
        Default system prompt
    """
    return load_prompt("default_system_prompt.txt")
