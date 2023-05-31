import numpy as np

class BanachSpace:
    def __init__(self, elements):
        self.elements = np.array(elements)

    def norm(self):
        return np.linalg.norm(self.elements)

    def scalar_multiply(self, scalar):
        return BanachSpace(self.elements * scalar)

    def add(self, other):
        if isinstance(other, BanachSpace):
            return BanachSpace(self.elements + other.elements)
        else:
            raise TypeError("Unsupported operand type for addition.")

    def subtract(self, other):
        if isinstance(other, BanachSpace):
            return BanachSpace(self.elements - other.elements)
        else:
            raise TypeError("Unsupported operand type for subtraction.")

    def dot_product(self, other):
        if isinstance(other, BanachSpace):
            return np.dot(self.elements, other.elements)
        else:
            raise TypeError("Unsupported operand type for dot product.")

    def __str__(self):
        return str(self.elements)
    
# Create two BanachSpace instances
A = BanachSpace([1, 2, 3])
B = BanachSpace([4, 5, 6])

# Perform operations on the instances
norm_A = A.norm()
scaled_A = A.scalar_multiply(2)
sum_AB = A.add(B)
dot_AB = A.dot_product(B)

# Print the results
print("A: ", A)
print("Norm of A: ", norm_A)
print("Scaled A: ", scaled_A)
print("Sum of A and B: ", sum_AB)
print("Dot product of A and B: ", dot_AB)
