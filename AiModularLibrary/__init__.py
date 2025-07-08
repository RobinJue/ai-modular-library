# AiModularLibrary - A modular library for AI system communication
# This library provides a unified interface for communicating with various AI systems
# like ChatGPT, Claude, and others.

__version__ = "1.0.0"
__author__ = "Your Name"

# Import and expose the main functions
from .main import ai_call_simple, ai_call_checked

__all__ = [
    'ai_call_simple',
    'ai_call_checked'
]
