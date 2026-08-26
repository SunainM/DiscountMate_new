"""Smoke test for the combined RAG + MCP chatbot agent.

Run from Backend/ml-service:
    python chatbots/evaluation/sample_agent_test.py
"""

import os
import sys


CURRENT_DIR = os.path.dirname(__file__)
ML_SERVICE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if ML_SERVICE_DIR not in sys.path:
    sys.path.insert(0, ML_SERVICE_DIR)

from chatbots.agents import DiscountMateAgent
from chatbots.mcp_tools import price_comparison, recipe_chat


class FakeRAG:
    def chat(self, session_id, user_query, top_k=5):
        return {
            "answer": "Use the Recipe RAG answer for chicken pasta.",
            "sources": [{"name": "Chicken Pasta", "score": 0.91}],
            "turns": 1,
            "limit_reached": False,
            "product_candidate_names": ["Chicken Breast"],
            "products_pending": True,
            "recipe_context_id": "ctx-demo",
        }


class FakeProductRepository:
    def search_products(self, **kwargs):
        return [{
            "product_id": "demo-coke-zero-2l",
            "product_name": "Coca-Cola Coke Zero 2L",
            "brand": "Coca-Cola",
            "pack_size": "2L",
            "category": "Soft drinks",
            "image_url": None,
            "score": 1.0,
        }]

    def get_product_details(self, product_id):
        return self.search_products(product_name="")[0]

    def get_prices(self, product_id, retailers=None):
        return [
            {"retailer": "Coles", "price": 3.20, "currency": "AUD"},
            {"retailer": "Woolworths", "price": 3.50, "currency": "AUD"},
        ]

    def compare_prices(self, product_id, retailers=None):
        prices = self.get_prices(product_id, retailers)
        return {
            "matched_product": self.search_products(product_name="")[0],
            "prices": prices,
            "cheapest": prices[0],
            "status": "success",
        }


def main():
    repo = FakeProductRepository()
    agent = DiscountMateAgent(
        tool_registry={
            "recipe_chat": recipe_chat.run,
            "compare_prices": lambda arguments: price_comparison.run(arguments, repository=repo),
        },
        rag_provider=lambda: FakeRAG(),
        enable_llm_planning=False,
    )

    recipe = agent.chat({
        "session_id": "demo-session",
        "message": "Show me a chicken pasta recipe",
    })
    assert recipe.success is True, recipe
    assert recipe.action == "recipe_chat", recipe
    assert recipe.data["recipe_context_id"] == "ctx-demo", recipe
    print("recipe_chat routed through combined agent")

    price = agent.chat({
        "session_id": "demo-session",
        "message": "Compare prices for Coke Zero 2L at Coles and Woolworths",
    })
    assert price.success is True, price
    assert price.action == "compare_prices", price
    assert price.data["cheapest"]["retailer"] == "Coles", price
    print("compare_prices routed through combined agent")

    invalid = agent.chat({"session_id": "", "message": ""})
    assert invalid.success is False, invalid
    assert invalid.error.code == "invalid_request", invalid
    print("invalid input returns structured validation error")

    print("Combined chatbot agent smoke test passed.")


if __name__ == "__main__":
    main()
