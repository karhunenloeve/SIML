import numpy as np
import math
from typing import List, Set, Dict, Tuple, Optional

def sample_dsphere(
	dimension: int, 
	amount: int, 
	radius: float = 1) -> np.ndarray:
	"""
	Creates uniform random sampling of a d-sphere.
	:param dimension: Integer as dimension of the embedding space.
	:param amount: Amount of sample points.
	:param radius: Radius of the d-sphere.
	:return: ndarray with data points.
	"""
	sphere = np.zeros(shape=(amount,dimension))

	for i in range(0, amount):
		# Generate (dimension) a certain amount of normally distributed random variales.
		x = np.random.normal(0,radius,dimension)
		# Distribute the points uniformly on a surface.
		# This works because the multivariate normal (dimension1,...,dimensionN)
		# is rotationally symmetric around the origin.
		y = np.sum(x**2) **(0.5)
		z = x/y

		for j in range(0,dimension):
			sphere[i][j] = z[j]

	return sphere


def sample_dball(
	dimension: int,
	amount: int,
	radius: float = 1) -> np.ndarray:
	"""
	This function samples from a d-ball by drop of coordinates.
	:param dimension: Integer as dimension of the embedding space.
	:param amount: Amount of sample points.
	:param radius: Radius of the d-ball.
	:return: ndarray with data points.
	"""
	# An array of d (dimension) normally distributed random variables.
	# In this case, the dimension is an integer.
	# This method is a result from https://www.sciencedirect.com/science/article/pii/S0047259X10001211.
	# The result has been proven by http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf.
	ball = np.zeros(shape=(amount,dimension))

	for i in range(0,amount):
		# Radius indicates the size of the ball.
		u = np.random.normal(0,radius,dimension + 2)
		norm = np.sum(u**2)**(0.5)
		u = u / norm
		x = u[0:dimension]

		for j in range(0,dimension):
			ball[i][j] = x[j]
		
	return ball


def sample_dtorus(
	dimension: int,
	amount: int,
	radii: list) -> np.ndarray:
	"""
	This function samples from a d-torus by rejection.
	"""
	try:
		if len(radii) > dimension:
			print("Take care, your radii list is longer then the amount of loops.")
			print("We selected only the first " + dimension + " entries.")


	except IndexError:
		print("The index of your radii list is out of range.")
	
print(sample_dball(3,100))
