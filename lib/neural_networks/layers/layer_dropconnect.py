from keras.layers import Dense, Wrapper
import keras.backend as K


class DropConnectDense(Dense):
    """
    Implementation of a drop connect layer.
    :param dense: A dense layer within a neural network.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize.
        :param args: None.
        :param kwargs: Pass a probability of dropout.
        """
        self.prob = kwargs.pop('prob', 0.5)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True
        super(DropConnectDense, self).__init__(*args, **kwargs)

    def call(self, x, mask=None):
        """
        Functions calls the layer.
        :param x: Datapoints.
        :param mask: None.
        :return: Dropout + activation on datapoints.
        """
        # Definition of dropout function in training phase.
        if 0. < self.prob < 1.:
            self.kernel = K.in_train_phase(K.dropout(self.kernel, self.prob), self.kernel)
            self.b = K.in_train_phase(K.dropout(self.b, self.prob), self.b)

        output = K.dot(x, self.W)
        if self.bias:
            output += self.b
        return self.activation(output)


class DropConnect(Wrapper):
    """
    Wrapper class for manipulation of drop connect layer.
    :param Wrapper: Keras Wrapper.
    """
    def __init__(self, layer, prob=1., **kwargs):
        """
        Initialize.
        :param layer: Previous layer.
        :param prob: Probability of dropout.
        :param kwargs: Further arguments (None).
        """
        self.prob = prob
        self.layer = layer
        super(DropConnect, self).__init__(layer, **kwargs)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True

    def build(self, input_shape):
        """
        Builds the layer by input_shape.
        :param input_shape: Input shape of the ndarray.
        :return: Output shape.
        """
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of a keras layer.
        :param input_shape: Input shape of the previous layer.
        :return: Output shape.
        """
        return self.layer.compute_output_shape(input_shape)

    def call(self, x):
        """
        Calls the layers functionality.
        :param x: Datapoints (numpy ndarray).
        :return: Processed data.
        """
        if 0. < self.prob < 1.:
            self.layer.kernel = K.in_train_phase(K.dropout(self.layer.kernel, self.prob), self.layer.kernel)
            self.layer.bias = K.in_train_phase(K.dropout(self.layer.bias, self.prob), self.layer.bias)
        return self.layer.call(x)