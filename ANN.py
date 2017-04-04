import numpy as np
import pandas
import csv
import math

class ANN():
	"""docstring for ANN"""
	def __init__(self, layers = layers, input = X, output = Y):
		self.layers = layers
		self.n_layers = len(layers)
