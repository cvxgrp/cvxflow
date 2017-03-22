
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class ProxSolver(object):
    def solve(self, max_iters=1000, epoch_iters=10, verbose=True):
        num_epochs = max_iters // epoch_iters

        graph = tf.Graph()
        with graph.as_default():
            # Initialize variables
            state = [tf.Variable(x) for x in self.initialize_graph()]

            def cond(*args):
                k, _ = args[0], args[1:]
                return k < epoch_iters

            def body(*args):
                k, state = args[0], args[1:]
                next_state = list(self.iterate(*state))
                return [k+1] + next_state

            next_state = tf.while_loop(cond, body, [0] + state)[1:]

            init_op = tf.global_variables_initializer()
            epoch_op = tf.group(
                *[x.assign(next_x) for x, next_x in zip(state, next_state)])

        with tf.Session(graph=graph) as sess:
            sess.run(init_op)
            for i in range(num_epochs):
                sess.run(epoch_op)
                if verbose:
                    print("iter " + str((i+1)*epoch_iters))

            state_np = sess.run(state)

        return state_np


class ADMM(ProxSolver):
    """Alternating direction method of multipliers.

    minimize    f(x) + g(z)
    subject to  Ax - Bz = 0
    """
    def __init__(self, prox_f, prox_g, variable_shape, A=None, B=None,
                 max_iters=100):
        self.prox_f = prox_f
        self.prox_g = prox_g
        self.variable_shape = variable_shape
        self.A = A
        self.B = B
        self.max_iters = max_iters

        if not isinstance(self.variable_shape, tuple):
            self.variable_shape = (
                self.variable_shape, self.variable_shape, self.variable_shape)
        if self.A is None:
            self.A = lambda x: x
        if self.B is None:
            self.B = lambda x: x

    def initialize_graph(self):
        self.prox_f.initialize_graph()
        self.prox_g.initialize_graph()
        x = tf.zeros(self.variable_shape[0])
        z = tf.zeros(self.variable_shape[1])
        u = tf.zeros(self.variable_shape[2])
        return x, z, u

    def iterate(self, x, z, u):
        x = self.prox_f(self.B(z) - u)
        z = self.prox_g(self.A(x) + u)
        u = u + self.A(x) - self.B(z)
        return x, z, u
