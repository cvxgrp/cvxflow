
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class ProxFunctionTest(test.TestCase):
    @property
    def _dtypes_to_test(self):
        return [dtypes.float32, dtypes.float64]

    def _verify(self, v, expected_x, **kwargs):
        for dtype in self._dtypes_to_test:
            with self.test_session() as sess:
                f = self._get_prox_function_for_dtype(dtype, **kwargs)
                x = sess.run(f(ops.convert_to_tensor(v, dtype=dtype)))
                self.assertEqual(dtype, x.dtype)
                self.assertAllClose(expected_x, x, atol=1e-5)
