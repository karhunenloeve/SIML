import numpy as np
from typing import List, Set, Dict, Tuple, Optional

def sample_dsphere(
	dimension: int, 
	amount: int, 
	radius: float = 1) -> np.ndarray:
	"""
	Creates uniform random sampling of a d-sphere.
	:param sphere: ndarray with data points.
	:param amount: Amount of sample points.
	:param radius: Radius of the d-sphere.
	:return: ndarray with dara points.
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
	radius: float = 1):
	pass
	
sample_dsphere(3,100)