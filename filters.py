import abc
from pathlib import Path
import numpy as np
import cv2
class Filter(metaclass = abc.ABCMeta):
    def __init__(self, filepath = Path):
        self.filepath = filepath
        
    def get_matrix_image(self):
        return cv2.imread(self.filepath)
        
    @abc.abstractmethod
    def filters(self)->np.array:
        pass
    






