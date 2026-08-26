"""MCP-style adapter around the existing Recipe RAG pipeline."""

from threading import Lock

from pydantic import ValidationError

from chatbots.schemas.recipes import RecipeChatArguments
from chatbots.schemas.tools import ToolError, ToolResponse


TOOL_NAME = "recipe_chat"

_RAG = None
_RAG_LOCK = Lock()


def run(arguments, rag_provider=None) -> ToolResponse:
    """Answer a cooking or recipe question through the existing RAG pipeline."""
    try:
        args = RecipeChatArguments(**(arguments or {}))
        rag = rag_provider() if rag_provider is not None else _default_rag()
        result = rag.chat(
            session_id=args.session_id,
            user_query=args.message,
            top_k=args.top_k,
        )
        return ToolResponse(success=True, tool=TOOL_NAME, data=result)
    except ValidationError as exc:
        return ToolResponse(
            success=False,
            tool=TOOL_NAME,
            error=ToolError(code="invalid_arguments", message=str(exc)),
        )
    except RuntimeError as exc:
        return ToolResponse(
            success=False,
            tool=TOOL_NAME,
            error=ToolError(code="rag_unavailable", message=str(exc)),
        )
    except Exception as exc:
        return ToolResponse(
            success=False,
            tool=TOOL_NAME,
            error=ToolError(code="tool_error", message=str(exc)),
        )


def _default_rag():
    global _RAG
    if _RAG is not None:
        return _RAG

    with _RAG_LOCK:
        if _RAG is None:
            from recipe_rag.rag_pipeline import RecipeRAG

            _RAG = RecipeRAG()
        return _RAG
