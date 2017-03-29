import tensorflow as tf

class IterativeSolver(object):
    """Abstract base class for iterative solvers."""
    def __init__(self, name=None):
        self.name = name or type(self).__name__

    @property
    def state(self):
        raise NotImplementedError

    def init(self):
        raise NotImplementedError

    def iterate(self, state):
        raise NotImplementedError

    def stop(self, state):
        raise NotImplementedError

    def solve(self):
        with tf.name_scope(self.name):
            with tf.name_scope("initialize"):
                init = self.init()

            def body(*args):
                with tf.name_scope("iterate"):
                    return self.iterate(self.state(*args))

            def cond(*args):
                with tf.name_scope("stop"):
                    return tf.logical_not(self.stop(self.state(*args)))

            return tf.while_loop(cond, body, init)
