"""Tool implementations exposed to chatbot agents."""

from chatbots.mcp_tools import price_comparison
from chatbots.mcp_tools import price_retrieval
from chatbots.mcp_tools import product_details
from chatbots.mcp_tools import product_search


TOOL_REGISTRY = {
    product_search.TOOL_NAME: product_search.run,
    product_details.TOOL_NAME: product_details.run,
    price_retrieval.TOOL_NAME: price_retrieval.run,
    price_comparison.TOOL_NAME: price_comparison.run,
}
