from keras import backend as K
from keras.optimizers import Optimizer
from keras.utils.generic_utils import get_from_module

import theano.tensor as T


class RMSprop_and_natGrad(Optimizer):
    """
    RMSProp optimizer with the capability to do natural gradient steps.
    It is recommended to leave the parameters of this optimizer
    at their default values (except the learning rate, which can be freely tuned).
    This optimizer is usually a good choice for recurrent neural networks.
    :param lr: float >= 0. Learning rate.
    :param rho: float >= 0.
    :param epsilon: float >= 0. Fuzzy factor.
    :param decay: float >= 0. Learning rate decay over each update.
    """

    def __init__(
        self, lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0, lr_natGrad=None, **kwargs
    ):
        """
        Initialize natural gradient descent with parameters.
        :param lr: Learning rate.
        :param rho: RMSProp parametrization (i.e. momentum).
        :param epsilon: RMSProp parametrization (i.e. momentum).
        :param decay: Decay term.
        :param lr_natGrad: Optional lr for natural gradient.
        :param kwargs: Additional arguments.
        """
        super(RMSprop_and_natGrad, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = K.variable(lr)

        if lr_natGrad is None:
            self.lr_natGrad = K.variable(lr)
        else:
            self.lr_natGrad = K.variable(lr_natGrad)

        self.rho = K.variable(rho)
        self.decay = K.variable(decay)
        self.inital_decay = decay
        self.iterations = K.variable(0.0)

    def get_updates(self, params, constraints, loss):
        """
        Update gradient step and parameter set.
        :param params: Parameterset to be optimized.
        :param constraints: List of constraints.
        :param loss: Loss functions of keras models.
        :return: A self update.
        """
        grads = self.get_gradients(loss, params)
        shapes = [K.get_variable_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators
        self.updates = []

        lr = self.lr
        if self.inital_decay > 0:
            lr *= 1.0 / (1.0 + self.decay * self.iterations)
            self.updates.append(K.update_add(self.iterations, 1))

        for param, grad, accum, shape in zip(params, grads, accumulators, shapes):

            if "natGrad" in param.name:
                if "natGradRMS" in param.name:
                    # Apply RMSprop rule to gradient before natural gradient step.
                    new_accum = self.rho * accum + (1.0 - self.rho) * K.square(grad)
                    self.updates.append(K.update(accum, new_accum))
                    grad = grad / (K.sqrt(new_accum) + self.epsilon)
                elif "unitaryAug" in param.name:
                    # We don't care about the accumulated RMS for the natural gradient step.
                    self.updates.append(K.update(accum, accum))

                # Do a natural gradient step.
                if "unitaryAug" in param.name:
                    # Unitary natural gradient step on augmented ReIm matrix.
                    j = K.cast(1j, "complex64")
                    A = K.cast(
                        K.transpose(param[: shape[1] / 2, : shape[1] / 2]), "complex64"
                    )
                    B = K.cast(
                        K.transpose(param[: shape[1] / 2, shape[1] / 2 :]), "complex64"
                    )
                    X = A + j * B
                    C = K.cast(
                        K.transpose(grad[: shape[1] / 2, : shape[1] / 2]), "complex64"
                    )
                    D = K.cast(
                        K.transpose(grad[: shape[1] / 2, shape[1] / 2 :]), "complex64"
                    )

                    # This part builds the skew Hermitian matrix.
                    # One can denote this as GX^H = CA^T + DB^T + jDA^T - jCB^T.
                    GXH = (
                        K.dot(C, K.transpose(A))
                        + K.dot(D, K.transpose(B))
                        + j * K.dot(D, K.transpose(A))
                        - j * K.dot(C, K.transpose(B))
                    )
                    Askew = GXH - K.transpose(T.conj(GXH))
                    I = K.eye(shape[1] / 2)
                    two = K.cast(2, "complex64")
                    CayleyDenom = I + (self.lr_natGrad / two) * Askew
                    CayleyNumer = I - (self.lr_natGrad / two) * Askew

                    # Multiplicative gradient step along Stiefel manifold equation.
                    Xnew = K.dot(
                        K.dot(T.nlinalg.matrix_inverse(CayleyDenom), CayleyNumer), X
                    )

                    # Convert to ReIm augmented form.
                    XnewRe = K.transpose(T.real(Xnew))
                    XnewIm = K.transpose(T.imag(Xnew))
                    new_param = K.concatenate(
                        (
                            K.concatenate((XnewRe, XnewIm), axis=1),
                            K.concatenate(((-1) * XnewIm, XnewRe), axis=1),
                        ),
                        axis=0,
                    )
                else:
                    # Do the usual RMSprop update using lr_natGrad as learning rate.
                    # Update accumulator.
                    new_accum = self.rho * accum + (1.0 - self.rho) * K.square(grad)
                    self.updates.append(K.update(accum, new_accum))
                    new_param = param - self.lr_natGrad * grad / (
                        K.sqrt(new_accum) + self.epsilon
                    )
            else:
                # Do the usual RMSprop update.
                # Applies changes to vector parametrizations.
                # Update accumulator.
                new_accum = self.rho * accum + (1.0 - self.rho) * K.square(grad)
                self.updates.append(K.update(accum, new_accum))
                new_param = param - lr * grad / (K.sqrt(new_accum) + self.epsilon)

            # Apply Constraints.
            if param in constraints:
                c = constraints[param]
                new_param = c(new_param)
            self.updates.append(K.update(param, new_param))
        return self.updates

    def get_config(self):
        """
        Function returns the configurations of the current running algorithms.
        :return: Dictionary.
        """
        config = {
            "lr": float(K.get_value(self.lr)),
            "lr_natGrad": float(K.get_value(self.lr_natGrad)),
            "rho": float(K.get_value(self.rho)),
            "epsilon": self.epsilon,
        }
        base_config = super(RMSprop_and_natGrad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
