"""
ConclaveSim: A simulation framework for papal elections using LLM agents.

This package provides the core framework for simulating papal conclaves
with AI-powered cardinal agents.
"""

__version__ = "0.1.0"

from .agents.base import Agent
from .environments.conclave_env import ConclaveEnv
from .config.manager import get_config

__all__ = ["Agent", "ConclaveEnv", "get_config"]
