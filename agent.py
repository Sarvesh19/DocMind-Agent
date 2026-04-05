import operator
import logging
from typing import TypedDict, Annotated, List, Literal, AsyncGenerator
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from vector_store import SupabaseVectorStore
from config import GROQ_API_KEY, GROQ_MODEL, OPENROUTER_API_KEY

logger = logging.getLogger(__name__)
_store: SupabaseVectorStore | None = None

def get_store() -> SupabaseVectorStore:
    global _store
    if _store is None:
        _store = SupabaseVectorStore()
    return _store

# ------------------ Tools ------------------
@tool
def search_knowledge_base(query: str) -> str:
    """Search uploaded documents for information relevant to the query. Always try this first."""
    results = get_store().similarity_search(query)
    if not results:
        return "No relevant information found in the knowledge base."
    parts = []
    for i, r in enumerate(results, 1):
        src = (r.get("metadata") or {}).get("source", "unknown")
        parts.append(f"[{i}] Source: {src} (score: {r.get('similarity', 0):.2f})\n{r['content']}")
    return "\n\n---\n\n".join(parts)

@tool
def web_search(query: str) -> str:
    """Search the web for current or external information not in the documents."""
    import requests
    try:
        resp = requests.get("https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}, timeout=8)
        data = resp.json()
        snippets = []
        if data.get("AbstractText"):
            snippets.append(data["AbstractText"])
        for t in data.get("RelatedTopics", [])[:4]:
            if isinstance(t, dict) and t.get("Text"):
                snippets.append(t["Text"])
        return "Web results:\n" + "\n\n".join(snippets) if snippets else f"No web results for: {query}"
    except Exception as e:
        return f"Web search error: {e}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +,-,*,/,**,sqrt,abs,round,min,max."""
    import math
    safe = {"__builtins__": {}, "sqrt": math.sqrt, "abs": abs, "round": round,
            "min": min, "max": max, "sum": sum, "log": math.log, "pi": math.pi, "e": math.e}
    try:
        return f"Result: {eval(expression, safe)}"
    except Exception as exc:
        return f"Calculation error: {exc}"

@tool
def summarise_document(source_name: str) -> str:
    """Generate a structured summary of a specific uploaded document by its filename."""
    store = get_store()
    results = store.similarity_search(f"main topics key points summary {source_name}", k=8, threshold=0.2)
    filtered = [r for r in results if source_name in (r.get("metadata") or {}).get("source", "")]
    if not filtered:
        return f"Document '{source_name}' not found. Please check the filename."
    combined = "\n\n".join(r["content"] for r in filtered[:6])
    return f"Content excerpts from '{source_name}':\n\n{combined}"

# SYSTEM_PROMPT = """You are DocMind, an expert Document Intelligence Agent. You help users understand documents and answer complex questions.

# Tools available:
# 1. search_knowledge_base — search uploaded docs (ALWAYS try first)
# 2. web_search — fetch live web information  
# 3. calculator — perform math calculations
# 4. summarise_document — summarise a specific file by name

# Rules:
# - Always search the knowledge base before using general knowledge or web
# - Cite sources like [Source: filename.pdf] when presenting document info
# - For complex questions, use multiple tools in sequence
# - Be thorough but concise. Use markdown formatting in responses.
# """

SYSTEM_PROMPT = """You are DocMind, an expert Document Intelligence Agent that helps users understand documents and answer complex questions.

**CRITICAL INSTRUCTION FOR WEB SEARCH:**
- **ALWAYS use the 'web_search' tool for ANY question about events, news, or situations that may have occurred after your knowledge cutoff in July 2024.**
- **This includes questions about the current state of conflicts, wars, political developments, or any time-sensitive topic.**
- **Do NOT rely on your internal knowledge for recent events. If you are unsure if your knowledge is current, you MUST use web_search.**

**Tool Usage Guidelines:**
1. `search_knowledge_base` - For searching uploaded documents (static, non-time-sensitive information)
2. `web_search` - For ANY question about recent events, news, or current situations (THIS IS YOUR DEFAULT FOR CURRENT EVENTS)
3. `calculator` - For performing mathematical calculations
4. `summarise_document` - For generating summaries of specific uploaded files

**Rules:**
- For questions about current events, conflicts, or recent news, ALWAYS start with `web_search`.
- Cite sources by mentioning the source domain (e.g., "According to Reuters...") when presenting information from web search results.
- Be thorough, accurate, and use markdown formatting in your responses.
- If a web search returns no relevant results, inform the user clearly.

Remember: Your internal knowledge is outdated. For anything that sounds like a current event, use web_search first."""

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

class RAGAgent:
    def __init__(self):
        # Use ChatOpenAI with Groq endpoint – no dependency conflict
        self.llm = ChatOpenAI(
            model=GROQ_MODEL,
            openai_api_key=GROQ_API_KEY,
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0.2,
            max_retries=2,
            streaming=True   # Enable native streaming from LLM
        )
        # self.llm = ChatOpenAI(
        #     model="openrouter/free",
        #     openai_api_key=OPENROUTER_API_KEY,
        #     openai_api_base="https://openrouter.ai/api/v1",
        #     temperature=0.2
        # )
        self.tools = [search_knowledge_base, web_search, calculator, summarise_document]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
        self.graph = self._build_graph()

    def _build_graph(self):
        def agent_node(state: AgentState):
            messages = state["messages"]
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: AgentState) -> Literal["tools", END]:
            last = state["messages"][-1]
            return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else END

        wf = StateGraph(AgentState)
        wf.add_node("agent", agent_node)
        wf.add_node("tools", self.tool_node)
        wf.add_edge(START, "agent")
        wf.add_conditional_edges("agent", should_continue)
        wf.add_edge("tools", "agent")
        return wf.compile()

    def invoke(self, user_input: str, history: List[BaseMessage] | None = None) -> str:
        messages = (history or []) + [HumanMessage(content=user_input)]
        result = self.graph.invoke({"messages": messages})
        final = result["messages"][-1]
        return final.content if hasattr(final, "content") else str(final)

    async def stream(self, user_input: str, history: List[BaseMessage] | None = None) -> AsyncGenerator[str, None]:
        """
        Async generator that yields tokens as they arrive from Groq.
        Uses the graph's astream_events to capture streaming output from the LLM.
        """
        messages = (history or []) + [HumanMessage(content=user_input)]
        # Add system prompt if needed
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        # Use astream_events to get token‑by‑token output
        async for event in self.graph.astream_events(
            {"messages": messages},
            version="v1",
            config={"run_id": None}  # optional
        ):
            # Look for token events from the LLM (when no tool calls are being made)
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content and not chunk.tool_calls:
                    yield chunk.content
            # Also handle final aggregated messages if needed (fallback)
            elif event["event"] == "on_chain_end" and event["name"] == "agent":
                # If streaming didn't produce tokens (e.g., tool-only flow), yield final
                output = event["data"]["output"].get("messages", [])[-1]
                if hasattr(output, "content") and output.content:
                    yield output.content