"""Smoothing strategies for n-gram language models."""

from .absolute_discounting import AbsoluteDiscounting
from .good_turing import GoodTuring
from .kneser_ney import KneserNey
from .laplace import Laplace

__all__ = ["Laplace", "GoodTuring", "AbsoluteDiscounting", "KneserNey"]
