


class AbsoluteValue(ProxFunction):
    """Absolute value.

    If lam is a scalar, lam*|x|
    If lam is a tuple, -lam[0]*neg(x) + lam[1]*pos(x)."""
    def __init__(self,
                 scale,
                 name="AbsoluteValue"):
        with ops.name_scope(name, values=[scale]):
            if isinstance(scale, tuple):
                assert len(scale) == 2
                self.scale_neg = ops.convert_to_tensor(scale[0], name="scale_neg")
                self.scale_pos = ops.convert_to_tensor(scale[1], name="scale_pos")
            else:
                self.scale_neg = ops.convert_to_tensor(scale, name="scale")
                self.scale_pos = self.scale_neg

            super(AbsoluteValue, self).__init__(
                graph_parents=[self.scale_neg, self.scale_pos],
                name=name)

    def _call(self, v):
        return (math_ops.maximum(v - self.scale_pos, 0) +
                math_ops.minimum(v + self.scale_neg, 0))
