"""Recipe chatbot schemas used by the combined agent."""

from pydantic import BaseModel, Field


class RecipeChatArguments(BaseModel):
    """Arguments for calling the existing Recipe RAG chatbot as a tool."""

    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
