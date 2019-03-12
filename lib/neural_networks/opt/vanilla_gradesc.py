import theano
import theano.tensor as T
import numpy as np

def clipped_gradients(gradients, gradient_clipping):
    """
    Clipping the gradient to guarantee more variety of directions.
    :param gradients: Gradients.
    :param gradient_clipping: Clipping values.
    :return: Clipped gradient.
    """
    clipped_grads = [T.clip(g, -gradient_clipping, gradient_clipping)
                     for g in gradients]
    return clipped_grads

def gradient_descent(learning_rate, parameters, gradients):
    """
    Vanilla gradient descent.
    :param learning_rate: Learning rate.
    :param parameters: Additional parameters.
    :param gradients: Respective Gradients.
    :return: Update.
    """
    updates = [(p, p - learning_rate * g) for p, g in zip(parameters, gradients)]
    return updates

def gradient_descent_momentum(learning_rate, momentum, parameters, gradients):
    """
    Vanilla gradient descent with momentum term.
    :param learning_rate: Learning rate.
    :param momentum: Momentum.
    :param parameters: Additional parameters.
    :param gradients: Gradients.
    :return: Update.
    """
    velocities = [theano.shared(np.zeros_like(p.get_value(),
                                              dtype=theano.config.floatX)) for p in parameters]

    updates1 = [(vel, momentum * vel - learning_rate * g)
                for vel, g in zip(velocities, gradients)]
    updates2 = [(p, p + vel) for p, vel in zip(parameters, velocities)]
    updates = updates1 + updates2
    return updates
