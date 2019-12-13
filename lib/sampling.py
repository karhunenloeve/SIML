import numpy as np
import math as mth
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


def sample_dtorus_cursed(
	dimension: int,
	amount: int,
	radii: list) -> np.ndarray:
	"""
	This function samples from a d-torus by rejection.
	The function is named cursed, because the curse of dimensionality leads to an exponential grouth in time.
	"""
	try:
		if len(radii) > dimension:
			print("Take care, your radii list is longer then the amount of loops.")
			print("We selected only the first " + dimension + " entries.")
		
		torus = np.zeros(shape=(amount,dimension))
		counter = amount

		while counter != 0:
			x = np.random.uniform(0,radii,dimension)

			for i in range(0,amount):
				def circle_sqrt(x,i):
					if i == 0:
						return x[0]**2 + x[1]**2
					else:
						return (mth.sqrt(x[i]**2 + circle_sqrt(x,i-1)) - radii[i])**2
				
				# One value possibly on the torus
				y = circle_sqrt(x,dimension-1)
				print(y)
				print(radii[0])
				# Check whether the coordinates really lie on the torus.
				# Note that this method is deeply cursed by the dimensions.
				if y == radii[0]**2:
					print("YES")
					for j in range(0,dimension):
						torus[i][j] = x[j]
						counter = counter - 1
				else:
					continue
		print(y)
		exit(1)

	except IndexError:
		print("The index of your radii list is out of range.")
	
print(sample_dtorus_cursed(2,10,[1,1]))
