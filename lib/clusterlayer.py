#!/usr/bin/env python
from sklearn.cluster import KMeans

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label.
    Example: model.add(ClusteringLayer(n_clusters=10))

    :param n_clusters: Number of clusters.
    :param weights: List of Numpy array with shape (n_clusters, n_features) witch represents the initial cluster centers.
    :param alpha: Degrees of freedom parameter in Student's T-distribution. Default to 1.0.

    Input shape: 2D tensor with shape: `(n_samples, n_features)`.
    Output shape: 2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        """
        Initialize Clustering layer.
        :param n_clusters: Number of cluster dimensions.
        :param weights: Initial weights (default None).
        :param alpha: Degree of freedom in T-distribution.
        :param kwargs: Additional optional arguments.
        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        """
        Input shape of the initial data manifold.
        :param input_shape: Data matrix as ndarray.
        :return: Weighted Clusters.
        """
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Call clusters.
        :param inputs: Input data.
        :param kwargs: additional arguments can be passed.
        :return: Normalization call for cluster.
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        """
        Compute the shape of the output manifold.
        :param input_shape: Input manifold of the data.
        :return: First position of input-shape, Number of clusters.
        """
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        """
        Returns the cluster configuration.
        :return: Dictionary.
        """
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))