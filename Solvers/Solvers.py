from Fields.Fields import Field
from Fields.Fields import ICScalarField
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

  def b_to_field(self):
    eid_to_n = self.field.kernel_list[0].eid_to_n
    n_to_eid = {}
    for eid in eid_to_n.keys():
      n = eid_to_n[eid]
      n_to_eid[n] = eid

    this_b = {}
    for _, nkey in enumerate(n_to_eid.keys()):
      eid = n_to_eid[nkey]
      n = nkey
      this_b[eid] = self.b[n]
    field = ICScalarField(m=self.field.global_mesh, name='bvec_'+self.field.name, volumeList=self.field.volumeList, fill_value=-1, ic_T = this_b)
    return field

  def diag_to_field(self):
    eid_to_n = self.field.kernel_list[0].eid_to_n
    n_to_eid = {}
    for eid in eid_to_n.keys():
      n = eid_to_n[eid]
      n_to_eid[n] = eid

    this_b = {}
    for _, nkey in enumerate(n_to_eid.keys()):
      eid = n_to_eid[nkey]
      n = nkey
      this_b[eid] = self.coeffs[n,n]
    field = ICScalarField(m=self.field.global_mesh, name='diag_'+self.field.name, volumeList=self.field.volumeList, fill_value=-1, ic_T = this_b)
    return field


  def solve(self):

    for it in range(self.iterations):
      print("SinglePhysicsSolver Field", self.field.name, " ~ Iteration", it)
      self.b *= 0.0
      self.coeffs *= 0.0

      # # try to set gradient term up if it exists
      try:
        print("\tUpdating gradient...")
        self.field.grad.updateGradient()
      except:
        print("Not updating gradient for variable", self.field.name, " - .updateGradient() does not exist.")

      # do boundary condition terms
      print("\tWriting coefficients...")
      for bc in self.field.bc_list:
        bc.setup_coeffs()
        self.coeffs += bc.coeffs
        self.b += bc.b

      # Now do kernels
      for kernel in self.field.kernel_list:
        kernel.setup_coeffs()
        self.coeffs += kernel.coeffs
        self.b += kernel.b

      # # Now solve
      print("\tSolving...")
      soln = scipySolve(self.coeffs, self.b)

      # # Fill dict with soln.
      for idx, key in enumerate(self.T.keys()):
        self.T[key] = soln[idx]
        self.field.T[key] = soln[idx]
