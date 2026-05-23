"""AgenticRec — From Pipeline to Council.

A lightweight agentic framework for search & recommendation.
"""
from .core import (
    AgentMessage,
    BaseAgent,
    Memory,
    Tool,
    ToolRegistry,
    Item,
    Decision,
    Trace,
)
from .llm import BaseLLM, MockLLM, DeepSeekLLM
from .agents import (
    RecallAgent,
    RankAgent,
    CollaborationAgent,
    RerankAgent,
    ExplainAgent,
    CriticAgent,
    OrchestratorAgent,
)
from .collab import AgentProfile, CollaborationReport, PreferenceVote
from .pipeline import AgenticPipeline, PipelineResult
from .tools import VectorTool, TagTool, FeatureTool, BizRuleTool, HotTool

__all__ = [
    "AgentMessage",
    "BaseAgent",
    "Memory",
    "Tool",
    "ToolRegistry",
    "Item",
    "Decision",
    "Trace",
    "BaseLLM",
    "MockLLM",
    "DeepSeekLLM",
    "RecallAgent",
    "RankAgent",
    "CollaborationAgent",
    "RerankAgent",
    "ExplainAgent",
    "CriticAgent",
    "OrchestratorAgent",
    "AgentProfile",
    "CollaborationReport",
    "PreferenceVote",
    "AgenticPipeline",
    "PipelineResult",
    "VectorTool",
    "TagTool",
    "FeatureTool",
    "BizRuleTool",
    "HotTool",
]

__version__ = "0.1.0"
