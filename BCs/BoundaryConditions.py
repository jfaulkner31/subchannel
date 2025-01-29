from gmesh_conv.meshing import mesh
from gmesh_conv.meshing import surface
from Fields.Fields import ScalarField
from Fields.Fields import Field
import numpy as np

class BoundaryCondition:
  def __init__(self, m: mesh, boundary_list: list, field: Field):
    # stuff from kernel
    self.eids = []
    self.eid_to_n = {} # key is the eid and value is the index of the matrix
    self.assign_elements(volumeList=field.volumeList) # makes eids list based on volumes
    self.n = len(self.eids)
    self.coeffs = np.zeros((self.n,self.n))
    self.b = np.zeros(self.n) # b part of Ax = b -> fill all zeros
    self.globalmesh = m

    # stuff for bc objects
    self.boundary_list = boundary_list # list of boundary ids
    self.field = field
    self.gradient_contribution = None
    self.fid_list = [] # list of face ids
    self.fid_to_eid = {} # dict[face_id] -> returns element id

    # assembly face and eid lists
    self.assemble_face_id_list() # fills self.fid_list

  def get_n(self, fid: int):
    return self.eid_to_n[self.fid_to_eid[fid]]

  def assemble_face_id_list(self):
    for boundary_id in self.boundary_list:
      boundary = self.globalmesh.boundaries[boundary_id]
      for face_id in boundary.face_ids:
        if face_id not in self.fid_list:
          self.fid_list.append(face_id)
    for fid in self.fid_list:
      self.fid_to_eid[fid] = self.globalmesh.faces[fid].own_id

  def get_gradient_contribution(self):
    return None # will error when called unless overridden

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




class DirichletBC(BoundaryCondition):
  def __init__(self, boundary_list: list, m: mesh, Gamma: float, field: ScalarField, value: float):
    super().__init__(boundary_list=boundary_list, m=m, field=field)
    self.Gamma = Gamma
    self.value = value

  def get_diags(self):
    # Gets and assigns diagonal coefficients
    # for _, fid in enumerate(self.fid_list):
    #   eid = self.globalmesh.faces[fid].own_id
    #   for fidx, this_gDiff in enumerate(self.globalmesh.gDiffs[eid]):
    #     # automatically adds the boundary terms as well as internal face terms since these terms are contained in gDiff
    #     if self.globalmesh.elements[eid].is_boundary[fidx]:
    #       fid = self.globalmesh.elements[eid].face_ids[fidx]
    #       mat_idx = self.get_n(fid)
    #       self.coeffs[mat_idx, mat_idx] += self.Gamma * this_gDiff

    for _, fid in enumerate(self.fid_list):
      eid = self.fid_to_eid[fid]
      fidx = self.globalmesh.elements[eid].face_ids.index(fid)
      this_gDiff = self.globalmesh.gDiffs[eid][fidx]
      mat_idx = self.get_n(fid)
      self.coeffs[mat_idx, mat_idx] += self.Gamma * this_gDiff
      if eid == 4524:
        pass
  def get_off_diags(self):
    # does nothing
    pass

  def get_b(self):
    for _, fid in enumerate(self.fid_list):
      eid = self.fid_to_eid[fid]
      fidx = self.globalmesh.elements[eid].face_ids.index(fid)
      grad_b = self.get_face_gradient(fid=fid, eid=eid)
      Tb = self.globalmesh.Tf[eid][fidx]
      gdiffB = self.globalmesh.gDiffs[eid][fidx]
      fluxVb = -self.Gamma * gdiffB * self.value - self.Gamma * np.dot(grad_b, Tb)
      self.b[self.get_n(fid)] -= fluxVb # subtracts fluxVb from b[n]

  def get_face_gradient(self, eid, fid):
    """
    Gets gradient at a boundary face.
    # TODO make the self.field.get_value(eid) term actually explicit in the matrix!
    # can reformulate the math sot hat this value is completely explicit
    """
    # get element geo weighting factor
    face_index_C = self.globalmesh.elements[eid].face_ids.index(fid)
    grad_b = ( self.value - self.field.get_value(eid) ) / self.globalmesh.elements[eid].d_Cf[face_index_C] * self.globalmesh.elements[eid].evec[face_index_C]
    return grad_b
    # return self.field.grad.gradient.get_value(eid)  # used to be grad_b but idk TODO check maybe?

    # TODO uFVM has return gradient of owner cell but i think that return grad_b is correct? idk
    # review uFVM implementation.

  def get_gradient_contribution(self, face_id: int, eid: int):
    # gets gradient contribution for this BC - e.g. phi_f * Surface_vector

    return self.value * self.globalmesh.faces[face_id].surface_vector

