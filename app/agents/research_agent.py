from app.services import llm
from app.prompts.research_agent_prompts import router_system_prompt,research_nod_system_prompt,orcastrator_nod_prompt
from app.utils.logger import logger

from pydantic import BaseModel, Field
from typing import TypedDict, List, Literal, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

import traceback
import time

from app.services.web_search.search_tools import _duckduckgo_search


# =========================
# ✅ Schemas
# =========================

class RouterDecision(BaseModel):
    need_to_research: bool
    mode: Literal["closed_book", "open_book", "hybrid"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_result_per_query: int = 5


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)
     

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class plan(BaseModel):
    blog_title : str
    audience : str
    tone : str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]





# =========================
# ✅ State
# =========================

class State(TypedDict):
    topic: str

    # router
    need_to_research: bool
    mode: Literal["closed_book", "open_book", "hybrid"]
    reason: str
    queries: List[str]
    recency_days: int

    # research
    evidence: List[EvidenceItem]
    plan : Optional[plan]


# =========================
# ✅ Router Node
# =========================

def router_node(state: State) -> dict:
    start_time = time.time()

    try:
        logger.info({
            "event": "router_node_started",
            "topic": state.get("topic")
        })

        decider = llm.with_structured_output(RouterDecision)

        decision = decider.invoke(
            [
                SystemMessage(content=router_system_prompt()),
                HumanMessage(content=f"Topic: {state['topic']}"),
            ]
        )

        logger.info({
            "event": "router_llm_response",
            "decision": decision.dict()
        })

        # recency logic
        if decision.mode == "open_book":
            recency_days = 7
        elif decision.mode == "hybrid":
            recency_days = 45
        else:
            recency_days = 3650

        duration = round(time.time() - start_time, 2)

        logger.success({
            "event": "router_node_success",
            "topic": state.get("topic"),
            "mode": decision.mode,
            "need_to_research": decision.need_to_research,
            "queries_count": len(decision.queries),
            "duration_sec": duration
        })

        return {
            "need_to_research": decision.need_to_research,
            "mode": decision.mode,
            "reason": decision.reason,
            "queries": decision.queries,
            "recency_days": recency_days,
        }

    except Exception as e:
        duration = round(time.time() - start_time, 2)

        logger.error({
            "event": "router_node_failed",
            "topic": state.get("topic"),
            "error": str(e),
            "duration_sec": duration
        })

        logger.error(traceback.format_exc())
        raise

## heplers 

def route_next(state: State) -> str:
    return "research" if state["need_to_research"] else "orchestrator"


# =========================
# ✅ Research Node
# =========================

def research_node(state: State) -> dict:
    start_time = time.time()

    try:
        logger.info({
            "event": "research_node_started",
            "topic": state.get("topic"),
            "queries_count": len(state.get("queries", []))
        })

        queries = (state.get("queries") or [])[:10]

        raw: List[dict] = []

        for q in queries:
            logger.info({
                "event": "search_query_started",
                "query": q
            })

            results = _duckduckgo_search(q, max_results=5)

            logger.info({
                "event": "search_query_completed",
                "query": q,
                "results_count": len(results)
            })

            raw.extend(results)

        if not raw:
            logger.warning({
                "event": "research_no_results",
                "topic": state.get("topic")
            })
            return {"evidence": []}

        logger.info({
            "event": "research_raw_collected",
            "total_results": len(raw)
        })

        # LLM extraction
        extractor = llm.with_structured_output(EvidencePack)

        logger.info({
            "event": "llm_extraction_started",
            "input_size": len(raw)
        })

        pack = extractor.invoke(
            [
                SystemMessage(content=research_nod_system_prompt()),
                HumanMessage(content=f"Raw results sample: {raw[:5]}")
            ]
        )

        duration = round(time.time() - start_time, 2)

        logger.success({
            "event": "research_node_success",
            "topic": state.get("topic"),
            "final_evidence_count": len(pack.evidence),
            "duration_sec": duration
        })

        dedup = {}
        for e in pack.evidence:
            if e.url:
                dedup[e.url] = e
        
        evidence = list(dedup.values())


        return {"evidence": pack.evidence}

    except Exception as e:
        duration = round(time.time() - start_time, 2)

        logger.error({
            "event": "research_node_failed",
            "topic": state.get("topic"),
            "error": str(e),
            "duration_sec": duration
        })

        logger.error(traceback.format_exc())
        raise

def orchestrator_node(state:State) -> dict:
    planner = llm.with_structured_output(plan)
    mode = state.get("mode","closed_book")
    evidence = state.get("evidence",[])

    forced_kind = 'news_roundup' if mode =="open_book" else None

    generated_plan = planner.invoke(
        [
         SystemMessage(content=orcastrator_nod_prompt()),
         HumanMessage(
             content=(
                      f"Topic:{state['topic']}\n"
                      f"Mode:{state['mode']}\n"
                      f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                      f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                      )
                      ),
                      ]
                      )

    if forced_kind:
        generated_plan.blog_kind = "news_roundup"

    return {"plan":generated_plan}







# =========================
# ✅ Build Graph
# =========================

# Nodes 

research_agent_graph = StateGraph(State)
research_agent_graph.add_node("router", router_node)
research_agent_graph.add_node("research", research_node)
research_agent_graph.add_node("orchestrator",orchestrator_node)




# Edges

research_agent_graph.add_edge(START, "router")
research_agent_graph.add_conditional_edges("router",route_next,{"research": "research", "orchestrator": "orchestrator"})
research_agent_graph.add_edge("research","orchestrator")
research_agent_graph.add_edge("orchestrator", END)

research_agent = research_agent_graph.compile()

result = research_agent.invoke({
    "topic": "AI in data science jobs"
})

print(result["plan"])