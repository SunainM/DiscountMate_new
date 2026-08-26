"""Combined RAG + MCP chatbot agent for DiscountMate."""

import json
import re
from typing import Any, Dict, Optional

from pydantic import ValidationError

from chatbots.mcp_tools import TOOL_REGISTRY
from chatbots.schemas.agent import (
    AgentResponse,
    AgentToolCall,
    ChatbotMessageRequest,
)
from chatbots.schemas.products import (
    PriceComparisonArguments,
    PriceRetrievalArguments,
    ProductDetailsArguments,
    ProductSearchArguments,
)
from chatbots.schemas.recipes import RecipeChatArguments
from chatbots.schemas.tools import ToolError


PLANNER_SYSTEM_PROMPT = """You are DiscountMate's tool planner.
Return only valid JSON with keys: tool, arguments, rationale.

Available tools:
- recipe_chat: recipes, cooking, meal planning, recipe ingredients, recipe steps.
- search_products: find grocery products by product_name, brand, pack_size, category.
- get_product_details: details for a known product_id.
- get_current_prices: current prices for a known product_id.
- compare_prices: compare retailer prices by product_id or product_name.
- clarification: ask for missing information.

Rules:
- Pick exactly one tool.
- Use recipe_chat for recipe/cooking requests even when ingredients may later map
  to products.
- Use compare_prices when the user asks for cheapest, price comparison, specials,
  deals, or where to buy a product cheaper.
- If a product-price request has no product_id and no product_name, choose
  clarification.
- Never invent a product_id.
- Arguments must match the selected tool only."""


PRODUCT_ARG_MODELS = {
    "recipe_chat": RecipeChatArguments,
    "search_products": ProductSearchArguments,
    "get_product_details": ProductDetailsArguments,
    "get_current_prices": PriceRetrievalArguments,
    "compare_prices": PriceComparisonArguments,
}


class DiscountMateAgent:
    """Routes a single chat message across Recipe RAG and product MCP tools."""

    def __init__(
        self,
        tool_registry: Optional[Dict] = None,
        rag_provider=None,
        llm_client=None,
        enable_llm_planning: bool = True,
    ):
        self.tool_registry = tool_registry or TOOL_REGISTRY
        self.rag_provider = rag_provider
        self.llm_client = llm_client
        self.enable_llm_planning = enable_llm_planning

    def chat(self, payload: Dict[str, Any]) -> AgentResponse:
        """Plan, validate, run one tool, and return a unified chatbot response."""
        try:
            request = ChatbotMessageRequest(**(payload or {}))
        except ValidationError as exc:
            return self._error_response(
                "clarification",
                "I need a session and message before I can help.",
                ToolError(code="invalid_request", message=str(exc)),
            )

        tool_call = self._plan(request)
        validation_error = self._validate_tool_call(tool_call)
        if validation_error:
            return self._error_response(
                tool_call.tool,
                "I could not safely run that chatbot action.",
                validation_error,
                tool_call=tool_call,
            )

        if tool_call.tool == "clarification":
            question = tool_call.arguments.get(
                "question",
                "Which product or recipe would you like help with?",
            )
            return AgentResponse(
                success=True,
                answer=question,
                action="clarification",
                tool_calls=[tool_call],
                needs_clarification=True,
                clarification_question=question,
            )

        response = self._run_tool(tool_call)
        response_dict = _model_dict(response)
        answer = self._answer_for_tool(tool_call, response_dict)
        needs_clarification, question = self._clarification_from_tool(response_dict)

        return AgentResponse(
            success=bool(response_dict.get("success")),
            answer=answer,
            action=tool_call.tool,
            tool_calls=[tool_call],
            data=response_dict.get("data") or {},
            needs_clarification=needs_clarification,
            clarification_question=question,
            error=response_dict.get("error"),
        )

    def _plan(self, request: ChatbotMessageRequest) -> AgentToolCall:
        tool_call = None
        if self.enable_llm_planning:
            tool_call = self._llm_plan(request)
        if tool_call is None:
            tool_call = self._heuristic_plan(request)
        return self._complete_tool_arguments(tool_call, request)

    def _llm_plan(self, request: ChatbotMessageRequest) -> Optional[AgentToolCall]:
        llm = self._get_llm_client()
        if llm is None:
            return None

        prompt = {
            "session_id": request.session_id,
            "message": request.message,
            "top_k": request.top_k,
            "context": _model_dict(request.context),
        }
        try:
            raw = llm.generate(
                PLANNER_SYSTEM_PROMPT,
                [{"role": "user", "content": json.dumps(prompt)}],
            )
            payload = _extract_json(raw)
            if not payload:
                return None
            return AgentToolCall(**payload)
        except Exception as exc:
            print(f"[DiscountMateAgent] planner fallback: {exc}")
            return None

    def _get_llm_client(self):
        if self.llm_client is not None:
            return self.llm_client
        try:
            from recipe_rag.rag_pipeline import LLMClient

            self.llm_client = LLMClient(temperature=0.0, max_tokens=350)
            return self.llm_client
        except Exception as exc:
            print(f"[DiscountMateAgent] LLM planner unavailable: {exc}")
            return None

    def _heuristic_plan(self, request: ChatbotMessageRequest) -> AgentToolCall:
        message = request.message.strip()
        lower = message.lower()
        product_id = request.context.product_id

        if self._is_recipe_request(lower):
            return AgentToolCall(
                tool="recipe_chat",
                arguments={
                    "session_id": request.session_id,
                    "message": message,
                    "top_k": request.top_k,
                },
                rationale="Recipe or cooking request.",
            )

        if self._is_price_request(lower):
            retailers = self._extract_retailers(lower)
            if product_id:
                return AgentToolCall(
                    tool="compare_prices",
                    arguments={
                        "product_id": product_id,
                        "retailers": retailers,
                    },
                    rationale="Price comparison for known product context.",
                )

            product_name = self._extract_product_name(message)
            if not product_name:
                return AgentToolCall(
                    tool="clarification",
                    arguments={
                        "question": "Which product would you like me to compare?",
                    },
                    rationale="Missing product name for price comparison.",
                )

            return AgentToolCall(
                tool="compare_prices",
                arguments={
                    "product_name": product_name,
                    "retailers": retailers,
                },
                rationale="Price comparison request.",
            )

        if product_id and re.search(r"\b(detail|info|about|price|cost)\b", lower):
            tool = "get_current_prices" if re.search(r"\b(price|cost)\b", lower) else "get_product_details"
            args = {"product_id": product_id}
            if tool == "get_current_prices":
                args["retailers"] = self._extract_retailers(lower)
            return AgentToolCall(tool=tool, arguments=args, rationale="Known product context.")

        product_name = self._extract_product_name(message)
        if product_name:
            return AgentToolCall(
                tool="search_products",
                arguments={"product_name": product_name, "limit": 5},
                rationale="Product lookup request.",
            )

        return AgentToolCall(
            tool="clarification",
            arguments={"question": "Do you want help with a recipe or a grocery product?"},
            rationale="No clear tool intent.",
        )

    def _validate_tool_call(self, tool_call: AgentToolCall) -> Optional[ToolError]:
        if tool_call.tool == "clarification":
            return None
        if tool_call.tool not in self.tool_registry:
            return ToolError(
                code="unknown_tool",
                message=f"Unknown chatbot tool: {tool_call.tool}",
            )

        model = PRODUCT_ARG_MODELS.get(tool_call.tool)
        if model is None:
            return ToolError(
                code="unknown_schema",
                message=f"No validation schema for chatbot tool: {tool_call.tool}",
            )

        try:
            validated_args = model(**(tool_call.arguments or {}))
            tool_call.arguments = _model_dict(validated_args)
        except ValidationError as exc:
            return ToolError(code="invalid_arguments", message=str(exc))
        return None

    def _complete_tool_arguments(
        self,
        tool_call: AgentToolCall,
        request: ChatbotMessageRequest,
    ) -> AgentToolCall:
        arguments = dict(tool_call.arguments or {})
        if tool_call.tool == "recipe_chat":
            arguments.setdefault("session_id", request.session_id)
            arguments.setdefault("message", request.message)
            arguments.setdefault("top_k", request.top_k)
        elif tool_call.tool in ("get_product_details", "get_current_prices", "compare_prices"):
            if request.context.product_id:
                arguments.setdefault("product_id", request.context.product_id)
        elif tool_call.tool == "search_products":
            if not arguments.get("product_name"):
                product_name = self._extract_product_name(request.message)
                if product_name:
                    arguments["product_name"] = product_name
            arguments.setdefault("limit", 5)

        if tool_call.tool == "compare_prices":
            if not arguments.get("product_id") and not arguments.get("product_name"):
                product_name = self._extract_product_name(request.message)
                if product_name:
                    arguments["product_name"] = product_name
            arguments.setdefault("retailers", self._extract_retailers(request.message.lower()))

        tool_call.arguments = arguments
        return tool_call

    def _run_tool(self, tool_call: AgentToolCall):
        if tool_call.tool == "recipe_chat":
            return self.tool_registry[tool_call.tool](
                tool_call.arguments,
                rag_provider=self.rag_provider,
            )
        return self.tool_registry[tool_call.tool](tool_call.arguments)

    def _answer_for_tool(self, tool_call: AgentToolCall, response: Dict[str, Any]) -> str:
        if not response.get("success"):
            error = response.get("error") or {}
            return error.get("message") or "That chatbot action failed."

        data = response.get("data") or {}
        if tool_call.tool == "recipe_chat":
            return data.get("answer") or "I found a recipe response."
        if tool_call.tool == "compare_prices":
            return self._price_comparison_answer(data)
        if tool_call.tool == "search_products":
            products = data.get("products") or []
            if not products:
                return "I could not find matching products."
            names = [p.get("product_name") for p in products[:5] if p.get("product_name")]
            return "I found these matching products: " + ", ".join(names)
        if tool_call.tool == "get_product_details":
            product = data.get("product") or {}
            name = product.get("product_name") or "that product"
            pack = product.get("pack_size")
            brand = product.get("brand")
            parts = [name]
            if brand:
                parts.append(f"brand: {brand}")
            if pack:
                parts.append(f"pack size: {pack}")
            return ". ".join(parts) + "."
        if tool_call.tool == "get_current_prices":
            prices = data.get("prices") or []
            if not prices:
                return "I could not find current prices for that product."
            formatted = [
                f"{p.get('retailer')}: ${float(p.get('price')):.2f}"
                for p in prices
                if p.get("retailer") and p.get("price") is not None
            ]
            return "Current prices are " + ", ".join(formatted) + "."
        return "Done."

    def _price_comparison_answer(self, data: Dict[str, Any]) -> str:
        status = data.get("status")
        question = data.get("clarification_question")
        if status == "needs_clarification" and question:
            return question
        if status == "not_found":
            return question or "I could not find that product in DiscountMate data."
        if status == "no_prices":
            return question or "I found the product, but no current prices are available."

        product = data.get("matched_product") or {}
        cheapest = data.get("cheapest") or {}
        prices = data.get("prices") or []
        if not cheapest:
            return "I found the product, but no current prices are available."

        name = product.get("product_name") or "that product"
        answer = (
            f"The cheapest current price for {name} is "
            f"${float(cheapest['price']):.2f} at {cheapest['retailer']}."
        )
        other_prices = [
            f"{p.get('retailer')} ${float(p.get('price')):.2f}"
            for p in prices[1:4]
            if p.get("retailer") and p.get("price") is not None
        ]
        if other_prices:
            answer += " Other prices: " + ", ".join(other_prices) + "."
        return answer

    def _clarification_from_tool(self, response: Dict[str, Any]):
        data = response.get("data") or {}
        question = data.get("clarification_question")
        return bool(question), question

    def _is_recipe_request(self, lower: str) -> bool:
        return bool(re.search(
            r"\b(recipe|cook|cooking|meal|dinner|lunch|breakfast|ingredient|"
            r"ingredients|make|prepare|bake|roast|fry|dish|serving|servings)\b",
            lower,
        ))

    def _is_price_request(self, lower: str) -> bool:
        return bool(re.search(
            r"\b(price|prices|cheapest|compare|comparison|cost|deal|deals|"
            r"special|specials|discount|buy)\b",
            lower,
        ))

    def _extract_retailers(self, lower: str):
        retailers = []
        for retailer in ("coles", "woolworths", "iga"):
            if retailer in lower:
                retailers.append(retailer)
        return retailers

    def _extract_product_name(self, message: str) -> str:
        text = message.strip()
        text = re.sub(r"\b(at|from)\s+(coles|woolworths|iga)\b", " ", text, flags=re.I)
        text = re.sub(
            r"\b(can you|could you|please|show me|find me|find|search|look up|"
            r"compare|price|prices|cheapest|current|cost|deal|deals|specials?|"
            r"for|of|the|a|an|product|products|grocery|groceries|where to buy)\b",
            " ",
            text,
            flags=re.I,
        )
        text = re.sub(r"[^A-Za-z0-9 .'-]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" .")
        return text[:120]

    def _error_response(
        self,
        action,
        answer: str,
        error: ToolError,
        tool_call: Optional[AgentToolCall] = None,
    ) -> AgentResponse:
        return AgentResponse(
            success=False,
            answer=answer,
            action=action,
            tool_calls=[tool_call] if tool_call else [],
            error=error,
        )


def _model_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return model


def _extract_json(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except ValueError:
        return {}
