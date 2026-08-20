"""MCP-style product details tool."""

from pydantic import ValidationError

from chatbots.schemas.products import ProductDetailsArguments
from chatbots.schemas.tools import ToolError, ToolResponse
from chatbots.services.product_repository import (
    MongoProductRepository,
    ProductRepositoryError,
)


TOOL_NAME = "get_product_details"


def run(arguments, repository=None) -> ToolResponse:
    """Return display-ready details for a known product."""
    try:
        args = ProductDetailsArguments(**(arguments or {}))
        repo = repository or MongoProductRepository()
        product = repo.get_product_details(args.product_id)
        if not product:
            return ToolResponse(
                success=False,
                tool=TOOL_NAME,
                error=ToolError(
                    code="product_not_found",
                    message="No product matched the supplied product_id",
                    details={"product_id": args.product_id},
                ),
            )
        return ToolResponse(success=True, tool=TOOL_NAME, data={"product": product})
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
