#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from numpy import genfromtxt

class SOM(object):
    def __init__(self, x, y, input_dim, learning_rate, radius, num_iter=111):
        """
        Initialization of the Kohonen feature map properties.
        :param x: Value.
        :param y: Target.
        :param input_dim: Dimension of the dataset i.e. columns.
        :param learning_rate: Learning rate of the feature map.
        :param radius: Learning function radius.
        :param num_iter: Amount of iterations.
        """
        self._x = x
        self._y = y
        self._learning_rate = float(learning_rate)
        self._radius = float(radius)
        self._num_iter = num_iter
        self._graph = tf.Graph()

        with self._graph.as_default():
            # Initialize graph.
            # Initializing variables and placeholders.
            self._weights = tf.Variable(tf.random_normal([x * y, input_dim]))
            self._locations = self._generate_index_matrix(x, y)
            self._input = tf.placeholder("float", [input_dim])
            self._iter_input = tf.placeholder("float")

            # Calculating BMU.
            input_matix = tf.stack([self._input for i in range(x * y)])
            distances = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self._weights, input_matix), 2), 1))
            bmu = tf.argmin(distances, 0)

            # Get BMU location.
            mask = tf.pad(tf.reshape(bmu, [1]), np.array([[0, 1]]))
            size = tf.cast(tf.constant(np.array([1, 2])), dtype=tf.int64)
            bmu_location = tf.reshape(tf.slice(self._locations, mask, size), [2])

            # Calculate learning rate and radius.
            decay_function = tf.subtract(1.0, tf.div(self._iter_input, self._num_iter))
            _current_learning_rate = tf.multiply(self._learning_rate, decay_function)
            _current_radius = tf.multiply(self._radius, decay_function)

            # Adapt learning rate to each neuron based on position.
            bmu_matrix = tf.stack([bmu_location for i in range(x * y)])
            bmu_distance = tf.reduce_sum(tf.pow(tf.subtract(self._locations, bmu_matrix), 2), 1)
            neighbourhood_func = tf.exp(
                tf.negative(tf.div(tf.cast(bmu_distance, "float32"), tf.pow(_current_radius, 2))))
            learning_rate_matrix = tf.multiply(_current_learning_rate, neighbourhood_func)

            # Update all the weights.
            multiplytiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_matrix, np.array([i]), np.array([1])), [input_dim])
                for i in range(x * y)])
            delta = tf.multiply(
                multiplytiplier,
                tf.subtract(tf.stack([self._input for i in range(x * y)]), self._weights))

            new_weights = tf.add(self._weights, delta)
            self._training = tf.assign(self._weights, new_weights)

            # Initialize session and run it.
            self._sess = tf.Session()
            initialization = tf.global_variables_initializer()
            self._sess.run(initialization)

    def train(self, input_vects):
        """
        Training function using input vectors as numpy-ndarray.
        :param input_vects: Dataset as numpy ndarray.
        :return: Nothing.
        """
        for iter_no in range(self._num_iter):
            for input_vect in input_vects:
                self._sess.run(self._training,
                               feed_dict={self._input: input_vect,
                                          self._iter_input: iter_no})

        self._centroid_matrix = [[] for i in range(self._x)]
        self._weights_list = list(self._sess.run(self._weights))
        self._locations = list(self._sess.run(self._locations))

        for i, loc in enumerate(self._locations):
            self._centroid_matrix[loc[0]].append(self._weights_list[i])

    def map_input(self, input_vectors):
        """
        Numpy ndarray as input data.
        :param input_vectors: Maps an input-vector to a certain session specific net.
        :return: Clustering points.
        """
        return_value = []
        for vect in input_vectors:
            min_index = min([i for i in range(len(self._weights_list))],
                            key=lambda x: np.linalg.norm(vect - self._weights_list[x]))
            return_value.append(self._locations[min_index])
        return return_value

    def _generate_index_matrix(self, x, y):
        """
        Generates index matrix for easier computation.
        :param x: Input.
        :param y: Target.
        :return: Numpy ndarray.
        """
        return tf.constant(np.array(list(self._iterator(x, y))))

    def _iterator(self, x, y):
        """
        Iterator over index matrix.
        :param x: Input.
        :param y: Target.
        :return: Numpy ndarray.
        """
        for i in range(x):
            for j in range(y):
                yield np.array([i, j])


som = SOM(6, 6, 9, 0.5, 0.5, 100)
data = genfromtxt('../data/abalone.csv', delimiter=',')
som.train(data)