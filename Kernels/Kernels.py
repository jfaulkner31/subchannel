import numpy as np
from Fields.Fields import ScalarField
from gmesh_conv.meshing import mesh
from Fields.Fields import Field
from Fields.Fields import ZeroScalarField
from Fields.Fields import VectorField
import matplotlib.pyplot as plt

class Kernel:
  """
  Sets up a Kernel object based on the volumeList we pass in.
  """
  def __init__(self, volumeList: list, m: mesh):
    self.eids = []
    self.eid_to_n = {} # key is the eid and value is the index of the matrix
    self.assign_elements(volumeList=volumeList) # makes eids list based on volumes
    self.n = len(self.eids)
    # TODO need to make a dict where key is the element id and the value is the index in the mesh
    self.coeffs = np.zeros((self.n,self.n))
    self.b = np.zeros(self.n) # b part of Ax = b -> fill all zeros
    self.globalmesh = m

    del volumeList # free volume list from memory

  def setup_coeffs(self):
    self.coeffs *= 0.0 # reset
    self.b *= 0.0 # reset
    self.get_diags()
    self.get_off_diags()
    self.get_b()

  def get_diags(self):
    pass
  def get_off_diags(self):
    pass
  def get_b(self):
    pass

  def assign_elements(self, volumeList):
    idx = int(0)
    for v in volumeList:
      for eid in v.elements:
        self.eids.append(eid)
        self.eid_to_n[eid] = idx
        idx += 1

  def plot_coeffs(self, coeff_range: int):
    """
    Plots the matrix as a grid where each cell's intensity represents the magnitude of the value in the matrix.
    """
    # Create the plot
    plt.figure(figsize=(8, 8))

    # Use imshow to plot the matrix
    # `cmap` determines the color map and `interpolation` smooths the grid (or keeps it pixelated)
    plt.imshow(np.abs(self.coeffs[0:coeff_range, 0:coeff_range]), cmap='viridis', interpolation='nearest')

    # Add color bar to show the mapping of values to colors
    plt.colorbar(label='Magnitude')

    # Add gridlines
    plt.grid(visible=True, which='both', color='black', linestyle='-', linewidth=0.5)

    # Add labels to the axes
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    # Show the plot
    plt.show()

class DiffusionKernel(Kernel):
  def __init__(self, field: ScalarField, volumeList: list, Gamma: float, orthogonalityApproach: str, m: mesh):
    super().__init__(volumeList=volumeList, m=m)
    self.Gamma = self.set_gamma(Gamma)
    self.orthogonalityApproach = orthogonalityApproach
    self.field = field

    # geometric diffusion coefficients for neighbors and boundaries.
    self.gDiffs = self.globalmesh.gDiffs

    # orthogonal, nonorthogonal, and surface vector components - dict[element_id / eid][list of vectors]
    self.Ef = self.globalmesh.Ef
    self.Tf = self.globalmesh.Tf

    # setup coeffs for ths kernel
    self.setup_coeffs()

  def get_diags(self):
    # get aC coeffs
    for idx, eid in enumerate(self.eids):
      for fidx, this_gDiff in enumerate(self.gDiffs[eid]):
        # do only face terms
        if self.globalmesh.elements[eid].is_face[fidx]:
          mat_idx = self.eid_to_n[eid]
          self.coeffs[mat_idx, mat_idx] += self.Gamma * this_gDiff

  def get_off_diags(self):
    # get off diagonal coeffs
    for _, eid in enumerate(self.eids):
      for fidx , _ in enumerate(self.globalmesh.elements[eid].face_ids):

        # Do faces with neighbor cells
        if self.globalmesh.elements[eid].is_face[fidx]:

          # get idx of equation for this element
          mat_idx = self.eid_to_n[eid] # get matrix idx for this element

          # get idx for neighbbor
          neighbor_id = self.globalmesh.elements[eid].neighbor_ids[fidx]
          neigh_idx = self.eid_to_n[neighbor_id]

          # get gDiff
          this_gDiff = self.gDiffs[eid][fidx]

          self.coeffs[mat_idx, neigh_idx] += -1.0 * this_gDiff * self.Gamma

  def get_b(self):
    for _, eid in enumerate(self.eids):
      this_b = 0.0
      for fidx , face_id in enumerate(self.globalmesh.elements[eid].face_ids):

        # Do only faces with neighbor cells
        if self.globalmesh.elements[eid].is_face[fidx]:
          # get idx for neighbbor
          neighbor_id = self.globalmesh.elements[eid].neighbor_ids[fidx]

          # get gradient at faces from current fields gradient solution
          face_grad = self.field.grad.get_face_gradient(eid=eid, fid=face_id)

          # orthogonal vector Tf
          Tf = self.Tf[eid][fidx]

          # now add term to b:
          this_b += np.dot(Tf, face_grad)

      this_n = self.eid_to_n[eid]
      self.b[this_n] += this_b

  def setup_coeffs(self):
    return super().setup_coeffs()

  def set_gamma(self, Gamma: float):
    # changes gamma coefficient used in the kernel if needed
    return Gamma








