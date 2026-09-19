from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from dayframe.agent.nodes import (
    agent_node,
    cluster_node,
    load_photos,
    prefilter_node,
    route_after_cluster,
    write_calendar,
)
from dayframe.agent.state import AgentContext, DayframeState
from dayframe.agent.tools import TOOLS


def build_graph(checkpointer: Any, *, approval_mode: str = "auto") -> Any:
    g = StateGraph(DayframeState, context_schema=AgentContext)
    g.add_node("load_photos", load_photos)
    g.add_node("prefilter", prefilter_node)
    g.add_node("cluster", cluster_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("write_calendar", write_calendar)

    g.add_edge(START, "load_photos")
    g.add_edge("load_photos", "prefilter")
    g.add_edge("prefilter", "cluster")
    g.add_conditional_edges(
        "cluster",
        route_after_cluster,
        {"agent": "agent", "write_calendar": "write_calendar"},
    )
    g.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: "write_calendar"},
    )
    g.add_edge("tools", "agent")
    g.add_edge("write_calendar", END)

    interrupt = ["write_calendar"] if approval_mode == "review" else []
    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt)
