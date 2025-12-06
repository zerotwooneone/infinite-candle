from abc import ABC, abstractmethod
import numpy as np

class Effect(ABC):
    def __init__(self, config):
        self.config = config
        self.buffer = None # Will be init by render logic

    @abstractmethod
    def update(self, dt: float):
        """Update physics/state"""
        pass

    @abstractmethod
    def render(self, buffer: np.ndarray, mapper):
        """Draw to the buffer"""
        pass