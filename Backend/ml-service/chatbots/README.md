# Chatbots

This folder is for DiscountMate chatbot work, including the DL-06 LLM agent,
MCP-style tool definitions, structured JSON contracts, and chatbot-specific
orchestration code.

## Suggested Structure

- `agents/` - LangGraph LLM agent workflows and orchestration logic.
- `langflow/` - Visual prototype exports for reviewing and testing graph shape.
- `mcp_tools/` - Product search, product details, price retrieval, and price comparison tools.
- `schemas/` - Request and response contracts used by the chatbot and tools.
- `services/` - Infrastructure clients and repositories used by the tools.

The existing recipe chatbot code currently lives in `../recipe_rag/`. New
general chatbot work should be added here, then integrated with the Flask
service in `../app.py` when endpoints are ready.

## First DL-06-T9 Tools

- `search_products` finds candidate product records from the current MongoDB data source.
- `get_product_details` returns display-ready product metadata and current prices.
- `get_current_prices` returns current valid prices for a resolved product.
- `compare_prices` combines product matching and price retrieval for the first end-to-end price comparison action.

The LLM/agent layer calls these tools through their `run(arguments)` functions.
The tools return a shared `ToolResponse` envelope so the LangGraph workflow can
route success, errors, and clarification states consistently.

## DL-06-T12 Orchestration Workflow

`agents/langgraph_workflow.py` contains the production orchestration graph used
by `DiscountMateAgent`. It keeps the public `agent.chat(payload)` interface used
by Flask, but internally runs these nodes:

1. `validate_request` - validates the incoming session, message, top_k, and
   optional app context.
2. `plan_tool` - uses an LLM planner when available, then falls back to the
   deterministic router.
3. `validate_tool` - validates the selected MCP-style tool arguments with
   Pydantic schemas.
4. `clarify` - returns a structured clarification question when required.
5. `execute_tool` - calls the selected product, price, comparison, or recipe
   tool.
6. `compose_answer` - returns the unified chatbot response used by the API.

When `langgraph` is installed, the workflow compiles a real `StateGraph`. Local
smoke tests still run before dependency installation through an equivalent
fallback runner that executes the same node methods.

The LangFlow visual prototype is stored at:

```bash
chatbots/langflow/discountmate_chatbot_flow.json
```

Use it as a review and testing reference for the workflow shape before building
custom LangFlow components around the same Python node functions.

## Local Smoke Test

Run this from `Backend/ml-service`:

```bash
python chatbots/evaluation/sample_tool_test.py
```

The sample test uses fake data, so it verifies the tool contracts and price
comparison workflow without needing MongoDB credentials.

To test against the real MongoDB data source, run:

```bash
python chatbots/evaluation/mongo_tool_test.py --product-name "Coke Zero" --pack-size "2L"
```

This reads `MONGO_URI` and `MONGO_DB_NAME` from `Backend/.env` or the current
environment and does not print credentials.

## API Routes

Flask ML service routes:

- `POST /api/chatbot/chat`
- `POST /api/chatbot/tools/recipe-chat`
- `POST /api/chatbot/tools/search-products`
- `POST /api/chatbot/tools/product-details`
- `POST /api/chatbot/tools/current-prices`
- `POST /api/chatbot/tools/compare-prices`

Express proxy routes, mounted under `/api/ml`:

- `POST /api/ml/chatbot/chat`
- `POST /api/ml/chatbot/tools/recipe-chat`
- `POST /api/ml/chatbot/tools/search-products`
- `POST /api/ml/chatbot/tools/product-details`
- `POST /api/ml/chatbot/tools/current-prices`
- `POST /api/ml/chatbot/tools/compare-prices`

The combined chatbot route uses a validated agent planner to choose between
the existing Recipe RAG flow and the MCP-style product/price tools.

Example request:

```json
{
  "product_name": "milk",
  "limit": 5
}
```

The endpoints also accept an `arguments` wrapper if the future agent sends a
full tool envelope:

```json
{
  "arguments": {
    "product_name": "milk",
    "limit": 5
  }
}
```
