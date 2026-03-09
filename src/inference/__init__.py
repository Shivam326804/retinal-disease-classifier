"""Inference module initialization"""
from .predictor import Predictor
from .grad_cam import GradCAMVisualizer

__all__ = ['Predictor', 'GradCAMVisualizer']
