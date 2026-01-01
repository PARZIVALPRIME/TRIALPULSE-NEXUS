"""
LLM Configuration for TRIALPULSE NEXUS 10X
Phase 5.1: LLM Setup
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class LLMProvider(Enum):
    """Available LLM providers"""
    OLLAMA = "ollama"
    GROQ = "groq"


@dataclass
class OllamaConfig:
    """Ollama configuration"""
    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"  # Your installed model
    timeout: int = 120
    options: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.7,
        "top_p": 0.9,
        "num_ctx": 4096,
        "num_predict": 1024,
    })


@dataclass
class GroqConfig:
    """Groq configuration (backup/fast inference)"""
    api_key: Optional[str] = None
    model: str = "llama-3.1-8b-instant"
    timeout: int = 60
    max_tokens: int = 1024
    temperature: float = 0.7
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("GROQ_API_KEY")


@dataclass
class LLMConfig:
    """Main LLM configuration"""
    primary_provider: LLMProvider = LLMProvider.OLLAMA
    fallback_provider: Optional[LLMProvider] = LLMProvider.GROQ
    
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    
    max_retries: int = 3
    retry_delay: float = 1.0
    
    enable_cache: bool = True
    cache_ttl: int = 3600
    
    log_requests: bool = True
    log_responses: bool = False


SYSTEM_PROMPTS = {
    "default": """You are an AI assistant for clinical trial data management. 
You help with data quality, issue resolution, and regulatory compliance.
Always be precise, cite evidence, and prioritize patient safety.
Never make medical decisions - only provide data-driven recommendations.""",

    "diagnostic": """You are a diagnostic AI agent for clinical trial issues.
Your role is to:
1. Analyze data quality issues
2. Identify root causes
3. Form hypotheses with confidence levels
4. Suggest verification steps
Always express uncertainty and never claim certainty.""",

    "resolver": """You are a resolution AI agent for clinical trial issues.
Your role is to:
1. Recommend solutions based on issue type
2. Prioritize by impact and urgency
3. Estimate effort required
4. Consider regulatory requirements
Always provide actionable, specific recommendations.""",

    "forecaster": """You are a forecasting AI agent for clinical trials.
Your role is to:
1. Predict timelines with uncertainty bands
2. Estimate resource requirements
3. Project data quality trajectories
4. Identify risks to milestones
Always provide ranges, not point estimates.""",

    "communicator": """You are a communication AI agent for clinical trials.
Your role is to:
1. Draft clear, professional messages
2. Tailor tone to the recipient
3. Include relevant context
4. Suggest appropriate channels
Never send messages automatically - only draft for human review.""",

    "safety": """You are a safety-focused AI assistant for clinical trials.
Your role is to:
1. Flag potential safety signals
2. Prioritize SAE reconciliation
3. Ensure timely reporting
4. Support but never replace medical review
Patient safety is the highest priority."""
}


TASK_MODEL_MAPPING = {
    "simple_query": "llama3.1:8b",
    "complex_analysis": "llama3.1:8b",
    "summarization": "llama3.1:8b",
    "report_generation": "llama3.1:8b",
    "code_generation": "llama3.1:8b",
}


def get_config() -> LLMConfig:
    """Get LLM configuration with environment overrides"""
    config = LLMConfig()
    
    if os.getenv("OLLAMA_HOST"):
        config.ollama.host = os.getenv("OLLAMA_HOST")
    
    if os.getenv("OLLAMA_MODEL"):
        config.ollama.model = os.getenv("OLLAMA_MODEL")
    
    if os.getenv("GROQ_API_KEY"):
        config.groq.api_key = os.getenv("GROQ_API_KEY")
    
    if os.getenv("LLM_PRIMARY_PROVIDER"):
        provider = os.getenv("LLM_PRIMARY_PROVIDER").lower()
        if provider == "groq":
            config.primary_provider = LLMProvider.GROQ
    
    return config