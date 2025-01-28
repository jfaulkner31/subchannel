from Fields.Fields import Field
from BCs.BoundaryConditions import BoundaryCondition
from Kernels.Kernels import Kernel
import scipy
from scipy.linalg import solve as scipySolve

class SinglePhysicsSolver:
  def __init__(self, field: Field, iterations: int):
    """
    Solver for a single variable on a single mesh.
    """
    self.field = field
    self.coeffs = self.field.kernel_list[0].coeffs * 0.0 # initialize to 0.0
    self.b = self.field.kernel_list[0].b * 0.0 # initialize to 0.0
    self.T = self.field.T
    self.iterations = iterations
  def solve(self):
    for it in range(self.iterations):
      print("SinglePhysicsSolver Field", self.field.name, " ~ Iteration", it)
      self.b *= 0.0
      self.coeffs *= 0.0

      # try to set gradient term up if it exists
      # try:
      self.field.grad.updateGradient()
      # except:
        # print("Not updating gradient - not existing.")
        # pass

      # do boundary condition terms
      for bc in self.field.bc_list:
        bc.setup_coeffs()
        self.coeffs += bc.coeffs
        self.b += bc.b

      # Now do kernels
      for kernel in self.field.kernel_list:
        kernel.setup_coeffs()
        self.coeffs += kernel.coeffs
        self.b += bc.b

      # Now solve
      soln = scipySolve(self.coeffs, self.b)

      # Fill dict with soln.
      for idx, key in enumerate(self.T.keys()):
        self.T[key] = soln[idx]
        self.field.T[key] = soln[idx]
