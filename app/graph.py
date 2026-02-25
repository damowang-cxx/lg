# app/graph.py
from __future__ import annotations

from typing import TypedDict, Literal, Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

class ResearchState(TypedDict, total=False):
    topic: str
    mode: Literal["auto", "approval"]          # auto=全自动, approval=关键节点审批
    plan: Dict[str, Any]                       # 研究计划（结构化）
    evidence: List[Dict[str, Any]]             # 证据块（片段+定位+来源）
    assets: Dict[str, Any]                     # 输出资产（问题树/时间线/争议点）
    status: str                                # 便于前端展示状态

def align_node(state: ResearchState) -> Dict[str, Any]:
    """把用户 topic 变成结构化研究计划（第一版先用规则生成，后续再接闭源 API）。"""
    topic = state["topic"]
    plan = {
        "topic": topic,
        "questions": [
            f"{topic} 的核心概念是什么？",
            f"{topic} 的关键分歧/争议点有哪些？",
            f"{topic} 的时间线/阶段划分怎么做？",
        ],
        "sources": ["web", "local_notes", "papers"],
        "deliverables": ["issue_tree", "timeline", "controversy_map", "citations"],
    }
    return {"plan": plan, "status": "planned"}

def route_after_align(state: ResearchState) -> Literal["approve_plan", "retrieve"]:
    """根据模式决定是否走审批节点。"""
    return "approve_plan" if state.get("mode") == "approval" else "retrieve"

def approve_plan_node(state: ResearchState) -> Command[Literal["retrieve", "end"]]:
    """人类审批：暂停图，等外部给 approve/reject。"""
    payload = {
        "type": "approve_plan",
        "message": "请确认研究计划是否可执行（True=通过 / False=终止）",
        "plan": state["plan"],
    }
    decision = interrupt(payload)  # 暂停，外部可拿到 payload；resume 的值会回到 decision 变量里 :contentReference[oaicite:6]{index=6}

    if decision is True:
        return Command(goto="retrieve", update={"status": "approved"})
    else:
        return Command(goto="end", update={"status": "rejected"})

def retrieve_node(state: ResearchState) -> Dict[str, Any]:
    """检索：第一版用 stub 模拟；后续替换成 Web/学术/Notion 本地工具。"""
    plan = state["plan"]
    # 这里先生成 2 条“证据块”示例：真实实现时要保证可审计（定位信息+访问日期+hash）
    evidence = [
        {
            "source_type": "web",
            "title": "Example Source 1",
            "url": "https://example.com/a",
            "accessed_at": "2026-02-25",
            "snippet": "这是一段示例摘录（将来替换为真实网页/PDF片段）",
            "locator": {"type": "url+text", "hint": "paragraph:3"},
        },
        {
            "source_type": "local",
            "title": "Example Note 1",
            "doc_id": "notion:xxxx",
            "snippet": "这是一段本地笔记摘录（将来替换为 Notion/PDF 真实片段）",
            "locator": {"type": "doc+offset", "offset": 1200, "length": 80},
        },
    ]
    return {"evidence": evidence, "status": "retrieved"}

def synthesize_node(state: ResearchState) -> Dict[str, Any]:
    """综合：第一版用规则输出“研究资产骨架”；后续接闭源模型 API 生成高质量资产。"""
    topic = state["topic"]
    assets = {
        "issue_tree": {
            "root": topic,
            "children": ["定义与边界", "关键理论/学派", "争议点", "关键事件与指标"],
        },
        "timeline": [
            {"t": "T0", "event": "起源/背景（占位）"},
            {"t": "T1", "event": "关键转折（占位）"},
        ],
        "controversy_map": [
            {"claim": "观点A（占位）", "evidence_refs": [0]},
            {"claim": "观点B（占位）", "evidence_refs": [1]},
        ],
        "citations": [
            {"ref_id": 0, "title": state["evidence"][0]["title"], "locator": state["evidence"][0]["locator"]},
            {"ref_id": 1, "title": state["evidence"][1]["title"], "locator": state["evidence"][1]["locator"]},
        ],
    }
    return {"assets": assets, "status": "done"}

def build_graph(checkpointer):
    builder = StateGraph(ResearchState)

    builder.add_node("align", align_node)
    builder.add_node("approve_plan", approve_plan_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "align")
    builder.add_conditional_edges("align", route_after_align, {
        "approve_plan": "approve_plan",
        "retrieve": "retrieve",
    })
    builder.add_edge("retrieve", "synthesize")
    builder.add_edge("synthesize", END)

    # approve_plan 的 Command(goto="end") 需要指向 END
    builder.add_node("end", lambda state: state)
    builder.add_edge("end", END)

    return builder.compile(checkpointer=checkpointer)