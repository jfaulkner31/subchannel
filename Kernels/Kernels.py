import numpy as np
from Fields.Fields import ScalarField
from gmesh_conv.meshing import mesh

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



class DiffusionKernel(Kernel):
  def __init__(self, T: ScalarField, volumeList: list, Gamma: float, orthogonalityApproach: str, m: mesh):
    super().__init__(volumeList=volumeList, m=m)
    self.Gamma = self.set_gamma(Gamma)
    self.orthogonalityApproach = orthogonalityApproach

    # geometric diffusion coefficients for neighbors and boundaries.
    self.gDiffs = {} # dict[element_id][list of floats]

    # orthogonal, nonorthogonal, and surface vector components - dict[element_id / eid][list of vectors]
    self.Ef = {}
    self.Tf = {}
    self.calculateNonorthogonalComponents() # updates (non)orthogonal components of the surfaces for every element
    self.setup_coeffs()

  def get_diags(self):
    # get aC coeffs
    for idx, eid in enumerate(self.eids):
      for this_gDiff in self.gDiffs[eid] :
        # automatically adds the boundary terms as well as internal face terms since these terms are contained in gDiff
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
      for fidx , _ in enumerate(self.globalmesh.elements[eid].face_ids):
        # Do only faces with neighbor cells
        if self.globalmesh.elements[eid].is_face[fidx]:
          mat_idx = self.eid_to_n[eid] # get matrix idx for this element
          # get idx for neighbbor
          neighbor_id = self.globalmesh.elements[eid].neighbor_ids[fidx]
          neigh_idx = self.eid_to_n[neighbor_id]

  def setup_coeffs(self):
    return super().setup_coeffs()

  def set_gamma(self, Gamma: float):
    # changes gamma coefficient used in the kernel if needed
    return Gamma

  def calculateNonorthogonalComponents(self):
    # Decomposes Sf into Sf = Ef + Tf
    # This method is how Ef and Tf are computed.

    for eid in self.eids:
      this_Ef = []
      this_Tf = []
      this_gDiff = [] # geometric diffusion coeffs

      e = self.globalmesh.elements[eid] # element with eid
      evecs = e.evec # list of normal vectors for each surface

      # now for every surface calculate Ef Tf and Sf
      for idx, faceid in enumerate(e.face_ids):
        if e.is_owner[idx]: #
          multiplier = 1 # S poiints away from cell for owner cells
        else:
          multiplier = -1
        Sf = self.globalmesh.faces[faceid].surface_vector * multiplier
        area = self.globalmesh.faces[faceid].area
        evec = evecs[idx]

        if evec is not None:
          # if internal face and NOT a boundary

          if self.orthogonalityApproach == 'MCA':
            _ef = np.dot(evec, Sf) * evec
            this_Ef.append(_ef)
            this_Tf.append(Sf - _ef)
          elif self.orthogonalityApproach == 'OCA':
            _ef = area * evec
            this_Ef.append(_ef)
            this_Tf.append(Sf - _ef)
          elif self.orthogonalityApproach == 'ORA':
            top =  np.dot(Sf, Sf)
            bottom = np.dot(evec, Sf)
            _ef = top/bottom * evec
            this_Ef.append(_ef)
            this_Tf.append(Sf - _ef)
          else:
            raise Exception("Unknown orthogonality approach")

          # set geometric diffusion coeff while we are here

          if e.is_face[idx]: # for faces
            this_gDiff.append( np.abs(_ef) / e.d_CNb[idx])
          elif e.is_boundary[idx]: # for boundaries
            this_gDiff.append(  np.abs(_ef) / e.d_Cf[idx] )
          else:
            raise Exception("Neither a face or boundary, huh????")


        else: # if it is a boundary face
          raise Exception("evec should be assigned for all faces and boundaries.")

      # now that we iterated over all faces, add to dictionaries
      self.Ef[eid] = this_Ef
      self.Tf[eid] = this_Tf
      self.gDiffs[eid] = this_gDiff









