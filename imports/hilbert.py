import math
from .quaternion import Quaternion

class HilbertSpace:
    def __init__(self, values):
        self.values = [Quaternion(*v) if isinstance(v, (tuple, list)) else v for v in values]

    def inner_product(self, other):
        if isinstance(other, HilbertSpace):
            return sum((q1.conjugate() * q2).w for q1, q2 in zip(self.values, other.values))
        elif isinstance(other, Quaternion):
            return sum((q.conjugate() * other).w for q in self.values)
        else:
            raise TypeError("Unsupported operand type for inner product")

    def norm(self):
        return math.sqrt(self.inner_product(self))

    def scalar_multiply(self, scalar):
        return HilbertSpace([q * scalar for q in self.values])

    def add(self, other):
        return HilbertSpace([q1 + q2 for q1, q2 in zip(self.values, other.values)])

    def subtract(self, other):
        return HilbertSpace([q1 - q2 for q1, q2 in zip(self.values, other.values)])

    def dot_product(self, other):
        return self.inner_product(other)

    def __str__(self):
        return str([str(q) for q in self.values])
