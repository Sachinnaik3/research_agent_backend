from app.services.llm.ai_manager import llm, generate_image_bytes
from app.prompts.research_agent_prompts import (
    router_system_prompt,
    research_nod_system_prompt,
    orcastrator_nod_prompt,
    image_system_prompt
)
from app.utils.logger import logger

from pydantic import BaseModel, Field
from typing import TypedDict, List, Literal, Optional, Annotated
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

import traceback
import time
import operator

from app.services.web_search.search_tools import _duckduckgo_search
from pathlib import Path
import re


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
    goal: str
    bullets: List[str]
    target_words: int

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup", "comparison", "system_design"
    ] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]

class ImageSpec(BaseModel):
            placeholder: str
            filename: str
            alt: str
            caption: str
            prompt: str

class GlobalImagePlan(BaseModel):
            md_with_placeholders: str
            images: List[ImageSpec]        


# =========================
# ✅ State
# =========================

class State(TypedDict):
    topic: str
    need_to_research: bool
    mode: Literal["closed_book", "open_book", "hybrid"]
    reason: str
    queries: List[str]
    recency_days: int
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str


# =========================
# Router Node
# =========================

def router_node(state: State) -> dict:
    try:
        logger.info({"event": "router_start", "topic": state.get("topic")})

        decider = llm.with_structured_output(RouterDecision)

        decision = decider.invoke(
            [
                SystemMessage(content=router_system_prompt()),
                HumanMessage(content=f"Topic: {state['topic']}"),
            ]
        )

        recency_days = (
            7 if decision.mode == "open_book"
            else 45 if decision.mode == "hybrid"
            else 3650
        )

        logger.success({"event": "router_success", "mode": decision.mode})

        return {
            "need_to_research": decision.need_to_research,
            "mode": decision.mode,
            "reason": decision.reason,
            "queries": decision.queries,
            "recency_days": recency_days,
        }

    except Exception as e:
        logger.error({"event": "router_error", "error": str(e)})
        logger.error(traceback.format_exc())
        raise


def route_next(state: State) -> str:
    return "research" if state["need_to_research"] else "orchestrator"


# =========================
# Research Node
# =========================

def research_node(state: State) -> dict:
    try:
        logger.info({"event": "research_start"})

        raw = []
        for q in (state.get("queries") or [])[:10]:
            raw.extend(_duckduckgo_search(q, max_results=5))

        if not raw:
            return {"evidence": []}

        extractor = llm.with_structured_output(EvidencePack)

        pack = extractor.invoke(
            [
                SystemMessage(content=research_nod_system_prompt()),
                HumanMessage(content=f"Raw results sample: {raw[:5]}"),
            ]
        )

        dedup = {e.url: e for e in pack.evidence if e.url}

        logger.success({"event": "research_success", "count": len(dedup)})

        return {"evidence": list(dedup.values())}

    except Exception as e:
        logger.error({"event": "research_error", "error": str(e)})
        logger.error(traceback.format_exc())
        raise


# =========================
# Orchestrator Node
# =========================

def orchestrator_node(state: State) -> dict:
    try:
        logger.info({"event": "orchestrator_start"})

        planner = llm.with_structured_output(Plan)

        plan = planner.invoke(
            [
                SystemMessage(content=orcastrator_nod_prompt()),
                HumanMessage(content=f"Topic: {state['topic']}"),
            ]
        )

        logger.success({"event": "orchestrator_success", "tasks": len(plan.tasks)})

        return {"plan": plan}

    except Exception as e:
        logger.error({"event": "orchestrator_error", "error": str(e)})
        logger.error(traceback.format_exc())
        raise


# =========================
# FANOUT
# =========================

def fanout(state: State):
    assert state["plan"] is not None

    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


# =========================
# Worker Node
# =========================

def worker_node(payload: dict) -> dict:
    try:
        task = Task(**payload["task"])

        logger.info({"event": "worker_start", "task_id": task.id})

        section_md = llm.invoke(
            [
                SystemMessage(content="Write blog section"),
                HumanMessage(content=f"Section: {task.title}"),
            ]
        ).content.strip()

        logger.success({"event": "worker_success", "task_id": task.id})

        return {"sections": [(task.id, section_md)]}

    except Exception as e:
        logger.error({"event": "worker_error", "error": str(e)})
        logger.error(traceback.format_exc())
        raise


# =========================
# Merge Node
# =========================

def merge_content(state: State) -> dict:
    try:
        logger.info({"event": "merge_start"})

        plan = state["plan"]

        ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
        merged_md = f"# {plan.blog_title}\n\n" + "\n\n".join(ordered)

        logger.success({"event": "merge_success"})

        return {"merged_md": merged_md}

    except Exception as e:
        logger.error({"event": "merge_error", "error": str(e)})
        logger.error(traceback.format_exc())
        raise


# =========================
# Image Planning Node
# =========================

def deside_image(state: State) -> dict:
    try:
        logger.info({"event": "image_plan_start"})

        merged_md = state["merged_md"]
        plan = state["plan"]

        planner = llm.with_structured_output(GlobalImagePlan)

        image_plan = planner.invoke(
            [
                SystemMessage(content=image_system_prompt()),
                HumanMessage(
                    content=(
                        f"Topic: {state['topic']}\n"
                        f"Blog kind: {plan.blog_kind}\n\n"
                        f"{merged_md}"
                    )
                ),
            ]
        )

        logger.success({"event": "image_plan_success"})

        return {
            "md_with_placeholders": image_plan.md_with_placeholders,
            "image_specs": [i.model_dump() for i in image_plan.images],
        }

    except Exception as e:
        logger.error({"event": "image_plan_error", "error": str(e)})
        logger.error(traceback.format_exc())
        raise

def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state['plan']
    assert plan is not None

    md = state.get("md_with_placeholders") or state['merged_md']
    image_spec = state.get('image_specs') or []

    # if no image requested just write merged md
    if not image_spec:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final":md}
    
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_spec:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        if not out_path.exists():
            try:
                img_bytes = generate_image_bytes(spec['prompt'])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

            img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
            md = md.replace(placeholder, img_md)
        
    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}



# =========================
# ✅ REDUCER SUBGRAPH 
# =========================

reducer_graph = StateGraph(State)

reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", deside_image)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()


# =========================
# MAIN GRAPH
# =========================

g = StateGraph(State)

g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)

# ✅ subgraph used
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")

g.add_conditional_edges(
    "router",
    route_next,
    {"research": "research", "orchestrator": "orchestrator"},
)

g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout)

g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

research_agent = g.compile()





def generate_response(topic: str) -> str:
    try:
        result = research_agent.invoke({"topic": topic})

        final_output = result.get("final")

        if not final_output:
            final_output = result.get("merged_md", "No content generated")

        return final_output

    except Exception as e:
        logger.error({"event": "generate_response_failed", "error": str(e)})
        raise


# # =========================
# # RUN
# # =========================

# if __name__ == "__main__":
#     try:
#         logger.info({"event": "execution_start"})

#         result = research_agent.invoke(
#             {"topic": "Transformer architecture"}
#         )

#         logger.success({"event": "execution_success"})

#         print(result.get("final"))

#     except Exception as e:
#         logger.error({"event": "execution_failed", "error": str(e)})
#         logger.error(traceback.format_exc())