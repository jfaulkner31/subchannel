from gmesh_conv.meshing import mesh
from Kernels.Kernels import Kernel
from Fields.Fields import ScalarField
import numpy as np

class BoundaryCondition(Kernel):
  TODO need to make it so that field can add bc and pass self to it. that way the BC can
  TODO use field data in its calcs and then the field can use things like
  TODO the appropriate face flux or face value in its gradient computations
  TODO field will call things like setup coeffs
  def __init__(self, m: mesh, boundary_name_list: list, field: ScalarField):
    super().__init__(volumeList=field.volumeList, m=m)
    self.boundary_name_list = boundary_name_list
    self.field = field
    self.gradient_contribution = None
    self.fid_to_n = {} # dict[face_id] -> returns value of n - the index of the matrix



  def assemble_face_id_list(self):
    pass
  def get_gradient_contribution(self):
    pass
  def setup_coeffs(self):
    return super().setup_coeffs()
  def get_diags(self):
    return super().get_diags()
  def get_off_diags(self):
    return super().get_off_diags()
  def get_b(self):
    return super().get_b()

class DirichletBC(BoundaryCondition):
  def __init__(self, boundary_name_list: list, m: mesh, Gamma: float, field: ScalarField, value: float):
    super().__init__(boundary_name_list, m, field=field)
    self.Gamma = Gamma
    self.value = value

  def get_diags(self):
    # Gets and assigns diagonal coefficients
    for idx, eid in enumerate(self.eids):
      for fidx, this_gDiff in enumerate(self.gDiffs[eid]):
        # automatically adds the boundary terms as well as internal face terms since these terms are contained in gDiff
        if self.globalmesh.elements[eid].is_boundary[fidx]:
          mat_idx = self.eid_to_n[eid]
          self.coeffs[mat_idx, mat_idx] += self.Gamma * this_gDiff

  def get_off_diags(self):
    # does nothing
    pass

  def get_b(self):
    for idx, eid in enumerate(self.eids):
      for fidx, fid in enumerate(self.globalmesh.elements[eid].face_ids):
        if self.globalmesh.elements[eid].is_boundary[fidx]:
          grad_b = self.get_face_gradient(fid=fid, eid=eid)
          Tb = self.globalmesh.Tf[eid][fidx]
          gdiffB = self.globalmesh.gDiffs[eid][fidx]
          fluxVb = -self.Gamma * gdiffB * self.value - self.Gamma * np.dot(grad_b, Tb)
          self.b -= fluxVb

  def get_face_gradient(self, eid, fid):
    """
    Gets gradient at a boundary face.
    # TODO make the self.field.get_value(eid) term actually explicit in the matrix!
    # can reformulate the math sot hat this value is completely explicit
    """
    # get element geo weighting factor
    face_index_C = self.global_mesh.elements[eid].face_ids.index(fid)
    grad_b = ( self.value - self.field.get_value(eid) ) / self.globalmesh.elements[eid].d_Cf * self.globalmesh.elements[eid].evec[face_index_C]
    return grad_b

  def get_gradient_contribution(self, face_id: int, eid: int):
    # gets gradient contribution for this BC - e.g. phi_f * Surface_vector
    return self.value * self.globalmesh.faces[face_id].surface_vector

