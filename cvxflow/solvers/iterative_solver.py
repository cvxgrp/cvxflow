import tensorflow as tf

class IterativeSolver(object):
    """Abstract base class for iterative solvers."""
    def __init__(self, name=None):
        self.name = name or type(self).__name__

    def init(self):
        raise NotImplementedError

    def iterate(self, state):
        raise NotImplementedError

    def stop(self, state):
        raise NotImplementedError

    def _iterate(self, state):
        with tf.name_scope(self.name):
            with tf.name_scope("iterate"):
                return self.iterate(state)

    def _stop(self, state):
        with tf.name_scope(self.name):
            with tf.name_scope("stop"):
                return self.stop(state)

    def _init(self):
        with tf.name_scope(self.name):
            with tf.name_scope("init"):
                return self.init()

    def solve(self):
        return tf.while_loop(
            lambda *args: tf.logical_not(self._stop(self.state(*args))),
            lambda *args: self._iterate(self.state(*args)),
            self._init())

def run_epochs(sess, solver, epoch_iterations, status, profile=False):
    state = solver.state(*[tf.Variable(x) for x in solver._init()])
    next_state = tf.while_loop(
        lambda k, state: k < epoch_iterations,
        lambda k, state: (k+1, solver._iterate(state)),
        (0, state))[1]

    epoch_op = tf.group(
        *[var.assign(val) for var, val in zip(state, next_state)])

    init_op = tf.global_variables_initializer()
    stop_op = solver._stop(state)
    sess.run(init_op)
    while not sess.run(stop_op):
        if profile:
            run_options = config_pb2.RunOptions(
                trace_level=config_pb2.RunOptions.FULL_TRACE)
            run_metadata = config_pb2.RunMetadata()
            sess.run(epoch_op, run_metadata=run_metadata)

            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            filename = "/tmp/cvxflow_%s_epoch_%d.json" % (solver.name, i)
            with open(filename, "w") as f:
                f.write(ctf)
        else:
            sess.run(epoch_op)

        sess.run(epoch_op)
        status(state)

    return state
