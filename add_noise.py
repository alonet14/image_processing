from importlib.resources import path
from pathlib import Path 
import cv2
import numpy as np
import abc
from numpy.random import Generator
import matplotlib.pyplot as plt

class Noise(metaclass = abc.ABCMeta):
    def __init__(self, filepath: Path):
        self.filepath = filepath
        
    def get_matrix_image(self):
        return cv2.imread(str(self.filepath), cv2.IMREAD_GRAYSCALE)
    
    @abc.abstractmethod
    def noise(self)->np.array:
        pass
    
    def add_noise(self) ->np.array:
        noise_gen = self.noise()
        origin_image = self.get_matrix_image()
        noise_image = origin_image+noise_gen
        return noise_image
       
class GaussianNoise(Noise):
    def __init__(self, filepath, mean = 0, var = 0.1):
        super().__init__(filepath=filepath)
        self.mean = mean
        self.var = var
    
    def noise(self)->np.array:
        image = self.get_matrix_image()
        row, col, ch = image.shape
        sigma = self.var**0.5
        gauss = np.random.normal(self.mean, sigma, (row, col, ch))
        return gauss
  
class RayleighNoise(Noise):
    def __init__(self, filepath, scale):
        super().__init__(filepath=filepath)
        self.scale = scale
        
    def noise(self)->np.array:
        image = self.get_matrix_image()
        row, col, ch = image.shape
        rayleigh = np.random.rayleigh(self.scale, (row, col, ch))
        return rayleigh
    
class ExponentialNoise(Noise):
    def __init__(self, filepath, scale):
        super().__init__(filepath = filepath)
        self.scale = scale
        
    def noise(self)->np.array:
        image = self.get_matrix_image()
        row, col, ch = image.shape        
        exponent = Generator.exponential(self.scale, (row, col, ch))
        return exponent
    
class SaltAndPeperNoise(Noise):
    def __init__(self,
                 filepath: Path,
                 s_vs_p = 0.5,
                 amount = 0.004):
        super().__init__(filepath)
        self.s_vs_p = s_vs_p
        self.amount = amount
    
    def noise(self)->np.array:
        image = self.get_matrix_image()
        # row,col,ch = image.shape

        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(self.amount * image.size * self.s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[tuple(coords)] = 1
        
        # Pepper mode
        num_pepper = np.ceil(self.amount* image.size * (1. - self.s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[tuple(coords)] = 0
        return np.asarray(out)
     
class UniformNoise(Noise):
    def __init__(self,
                 filepath:Path,
                 a = 0.0, 
                 b = 1.0):
        super().__init__(filepath=filepath)
        self.a = a
        self.b = b
        
    def noise(self)->np.array:
        image = self.get_matrix_image()
        row, col, ch = image.shape
        uniform_gen = Generator.uniform(self.a, self.b, (row, col, ch))
        print(uniform_gen)
        return np.asarray(uniform_gen)
    


        