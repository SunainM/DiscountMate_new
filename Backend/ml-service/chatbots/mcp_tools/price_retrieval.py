"""MCP-style current price retrieval tool."""

from pydantic import ValidationError

from chatbots.schemas.products import PriceRetrievalArguments
from chatbots.schemas.tools import ToolError, ToolResponse
from chatbots.services.product_repository import (
    MongoProductRepository,
    ProductRepositoryError,
)


TOOL_NAME = "get_current_prices"


def run(arguments, repository=None) -> ToolResponse:
    """Return current valid retailer prices for a known product."""
    try:
        args = PriceRetrievalArguments(**(arguments or {}))
        repo = repository or MongoProductRepository()
        prices = repo.get_prices(args.product_id, args.retailers)
        return ToolResponse(
            success=True,
            tool=TOOL_NAME,
            data={"product_id": args.product_id, "prices": prices},
        )
    except ValidationError as exc:
        return ToolResponse(
            success=False,
            tool=TOOL_NAME,
            error=ToolError(code="invalid_arguments", message=str(exc)),
        )
    except ProductRepositoryError as exc:
        return ToolResponse(
            success=False,
            tool=TOOL_NAME,
            error=ToolError(code="repository_unavailable", message=str(exc)),
        )
    except Exception as exc:
        return ToolResponse(
            success=False,
            tool=TOOL_NAME,
            error=ToolError(code="tool_error", message=str(exc)),
        )
