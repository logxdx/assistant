def HumanMessage(content: str) -> dict:
    """Creates a human message dictionary for the chat model."""
    return {
        "role": "user",
        "content": content
    }

def SystemMessage(content: str) -> dict:
    """Creates a system message dictionary for the chat model."""
    return {
        "role": "system",
        "content": content
    }

def AssistantMessage(content: str) -> dict:
    """Creates an assistant message dictionary for the chat model."""
    return {
        "role": "assistant",
        "content": content
    }