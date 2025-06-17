"""
Agent mixins for the Conclave simulation.

This module provides behavior mixins that can be composed together to create
the full Agent functionality, improving maintainability and testability.
"""

from .voting import VotingMixin
from .discussion import DiscussionMixin
from .stance import StanceMixin
from .reflection import ReflectionMixin

__all__ = [
    'VotingMixin',
    'DiscussionMixin', 
    'StanceMixin',
    'ReflectionMixin'
]
