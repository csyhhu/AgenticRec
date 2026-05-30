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
from .gating import GateDecision, IntentGate
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
from .vector_backend import (
    ExternalVectorBackend,
    FaissVectorBackend,
    HashEmbedding,
    InMemoryVectorBackend,
    MilvusVectorBackend,
    VectorBackend,
    VectorHit,
)

__all__ = [
    "AgentMessage",
    "BaseAgent",
    "Memory",
    "Tool",
    "ToolRegistry",
    "Item",
    "Decision",
    "Trace",
    "GateDecision",
    "IntentGate",
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
    "ExternalVectorBackend",
    "FaissVectorBackend",
    "HashEmbedding",
    "InMemoryVectorBackend",
    "MilvusVectorBackend",
    "VectorBackend",
    "VectorHit",
]

__version__ = "0.1.0"
