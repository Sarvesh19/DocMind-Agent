import operator
import logging
from typing import TypedDict, Annotated, List, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from config import *

logger = logging.getLogger(__name__)

_store = None
def get_store():
    global _store
    if _store is None:
        from vector_store import SupabaseVectorStore
        _store = SupabaseVectorStore()
    return _store

# ── Tools (optimized for size limits) ─────────────────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the uploaded documents knowledge base.
    Returns at most 4 chunks, each truncated to 800 chars.
    """
    results = get_store().similarity_search(query, k=4)   # fewer chunks
    if not results:
        return "NO_RESULTS: No information found in the knowledge base for this query."
    parts = []
    for i, r in enumerate(results, 1):
        src = (r.get("metadata") or {}).get("source", "unknown")
        content = r['content']
        if len(content) > 800:
            content = content[:800] + "... [truncated]"
        parts.append(f"[Chunk {i}] Source: {src} | Similarity: {r.get('similarity', 0):.2f}\n{content}")
    return "\n\n---\n\n".join(parts)

@tool
def web_search(query: str) -> str:
    """
    Search the internet for current, real-time, or additional information.
    Use when KB has no results or user asks for latest news.
    """
    import requests
    try:
        resp = requests.get("https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}, timeout=10)
        data = resp.json()
        snippets = []
        if data.get("AbstractText"):
            snippets.append(f"Summary: {data['AbstractText']}")
        for t in data.get("RelatedTopics", [])[:4]:
            if isinstance(t, dict) and t.get("Text"):
                snippets.append(t["Text"])
        if snippets:
            return "Web search results:\n\n" + "\n\n".join(snippets)
        if data.get("Answer"):
            return f"Answer: {data['Answer']}"
        return f"No web results found for: {query}."
    except Exception as e:
        return f"Web search error: {e}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    import math
    safe = {"__builtins__": {}, "sqrt": math.sqrt, "abs": abs, "round": round,
            "min": min, "max": max, "sum": sum, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e, "pow": pow}
    try:
        return f"Result: {eval(expression, safe)}"
    except Exception as exc:
        return f"Calculation error: {exc}"

@tool
def summarise_document(source_name: str) -> str:
    """Retrieve and summarise a specific uploaded document."""
    store = get_store()
    results = store.similarity_search(
        f"main topics overview introduction background key points conclusions {source_name}",
        k=6, threshold=0.15
    )
    filtered = [r for r in results if source_name in (r.get("metadata") or {}).get("source", "")]
    if not filtered:
        filtered = results[:4]
    if not filtered:
        return f"Document '{source_name}' not found."
    combined = "\n\n".join(r["content"][:1000] for r in filtered[:6])
    return f"Document content from '{source_name}':\n\n{combined}"

# ── System prompt (shortened to save tokens) ──────────────────────────────────

SYSTEM_PROMPT = """You are DocMind, an AI that answers questions from uploaded documents.

Rules:
1. ALWAYS call search_knowledge_base first.
2. Use ONLY document content to answer. Do NOT call web_search if KB returned content.
3. Only call web_search if KB returns "NO_RESULTS" OR user asks for current events.
4. Cite sources: [Source: filename.pdf]
5. Be thorough and structured (use ## headers, bullet points, bold key terms).
"""

# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# ── RAGAgent with fallback on 429 ─────────────────────────────────────────────

class RAGAgent:
    def __init__(self):
        # Primary: Groq (fast, but rate‑limited)
        if GROQ_API_KEY:
            self.primary_llm = ChatOpenAI(
                model=GROQ_MODEL,
                openai_api_key=GROQ_API_KEY,
                openai_api_base="https://api.groq.com/openai/v1",
                temperature=0.3,
                max_retries=1,          # we handle retries manually
                request_timeout=30,
            )
        else:
            self.primary_llm = None

        # Fallback: OpenRouter free models (unlimited, slower)
        self.fallback_llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENROUTER_API_KEY or "sk-placeholder",
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=0.3,
            max_retries=2,
        )

        self.tools = [search_knowledge_base, web_search, calculator, summarise_document]
        self.tool_node = ToolNode(self.tools)
        self.graph = self._build_graph()

    def _call_llm_with_fallback(self, messages):
        """Try Groq; if 429 (rate limit), fallback to OpenRouter."""
        if self.primary_llm is None:
            return self.fallback_llm.bind_tools(self.tools).invoke(messages)

        llm_with_tools = self.primary_llm.bind_tools(self.tools)
        try:
            return llm_with_tools.invoke(messages)
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                logger.warning("Groq rate limit hit, falling back to OpenRouter free model")
                return self.fallback_llm.bind_tools(self.tools).invoke(messages)
            raise

    def _build_graph(self):
        def agent_node(state: AgentState):
            msgs = state["messages"]
            # Trim history to avoid token overflow
            if len(msgs) > 12:
                msgs = msgs[-12:]
            if not msgs or not isinstance(msgs[0], SystemMessage):
                msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs

            response = self._call_llm_with_fallback(msgs)
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

    async def stream(self, user_input: str, history: List[BaseMessage] | None = None):
        messages = (history or []) + [HumanMessage(content=user_input)]
        async for event in self.graph.astream({"messages": messages}, stream_mode="values"):
            last = event["messages"][-1]
            if isinstance(last, AIMessage) and last.content and not getattr(last, "tool_calls", None):
                yield last.content