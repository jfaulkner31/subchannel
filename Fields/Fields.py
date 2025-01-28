import numpy as np
from gmesh_conv.meshing  import mesh


class Field:
  def __init__(self, volumeList: list, m: mesh, name: str):
    self.volumeList = volumeList # a list of volume objects
    self.global_mesh = m
    self.name = name
    self.bc_list = [] # list of BoundaryCondition derived objects
    self.kernel_list = [] # list of kernels
    self.T = None
  def assign_bcs(self, bc_list: list):
    # assigns BoundaryCondition  objcets to self.bc_list
    # override this method later on in nearly all fields that use gradients.
    for bc in bc_list:
      self.bc_list.append(bc)
  def assign_kernels(self, kernel_list: list):
    for kernel in kernel_list:
      self.kernel_list.append(kernel)


class ICScalarField(Field):
  def __init__(self, volumeList: list, m: mesh, fill_value: float, ic_T={}, name='defaultField'):
    super().__init__(volumeList=volumeList, m=m, name=name)
    self.fill_value = fill_value
    self.ic_T = ic_T
    self.T, self.eids = self.assign_values()
    self.n = len(self.T) # size of field


  def assign_values(self):
    # fill with predetermined float
    if len(self.ic_T) == 0:
      _T = {}
      eids = []
      for v in self.volumeList:
        for eid in v.elements:
          _T[eid] = self.fill_value
          eids.append(eid)
    # fill with a field
    else:
      _T = {}
      eids = []
      for v in self.volumeList:
        for eid in v.elements:
          _T[eid] = self.ic_T[eid]
          eids.append(eid)
    return _T, eids

class ScalarField(Field):
  def __init__(self, initial_condition: Field, name: str, gradientTolerance=1e-6):
    """
    volumeList: List of volume objects that this ScalarField lives on.
    """
    super().__init__(volumeList=initial_condition.volumeList, m=initial_condition.global_mesh, name=name)
    self.gradientTolerance = gradientTolerance
    self.T = self.set_initial_condition(ic=initial_condition)
    self.eids = initial_condition.eids # copies element ids of the initial conditions
    # here we call boundary conditions and pass in self.


    # this field but on faces
    self.faceField = FaceScalarField(m=self.global_mesh, volumeList=self.volumeList, name=self.name+'_face')
    self.grad = None # gradient made in self.assign_bcs def

  def assign_bcs(self, bc_list):
    # first assign all bcs
    for bc in bc_list:
      self.bc_list.append(bc)
    # now make gradient
    self.grad = FVGradient(m=self.global_mesh, volumeList=self.volumeList, field=self, tolerance=self.gradientTolerance, bc_list = self.bc_list)

  def assign_kernels(self, kernel_list):
    return super().assign_kernels(kernel_list)

  def set_initial_condition(self, ic: Field):
    # we use the function to set the IC so that code creates a new copy -- self.T = ic.T would just pass a reference
    T = {}
    for key in ic.T.keys():
      T[key] = ic.T[key]
    return T

  def get_value(self, eid):
    return self.T[eid]

class VectorField(Field):
  def __init__(self, volumeList: list, m: mesh, name: str):
    super().__init__(volumeList=volumeList, m=m, name=name)
    self.T, self.eids = self.set_zero()

  def decompose_into_parts(self):
    """
    Splits the 3D vector into three scalar fields and returns them as a list.
    """
    fX = {}
    fY = {}
    fZ = {}
    for key in self.T.keys():
      fX[key] = self.T[key][0]
      fY[key] = self.T[key][1]
      fZ[key] = self.T[key][2]
    fX = ICScalarField(volumeList=self.volumeList, m=self.global_mesh, fill_value=-1, ic_T = fX, name=self.name+'_X')
    fY = ICScalarField(volumeList=self.volumeList, m=self.global_mesh, fill_value=-1, ic_T = fY, name=self.name+'_Y')
    fZ = ICScalarField(volumeList=self.volumeList, m=self.global_mesh, fill_value=-1, ic_T = fZ, name=self.name+'_Z')
    return [fX, fY, fZ]

  def set_zero(self):
    _T = {}
    eids = []
    for v in self.volumeList:
      for eid in v.elements:
        _T[eid] = np.array([0.0, 0.0, 0.0])
        eids.append(eid)
    return _T, eids

  def set_value(self, eid: int, vec: np.ndarray):
    for idx, v in enumerate(self.T[eid]):
      self.T[eid][idx] = vec[idx]

  def get_value(self, eid):
    return self.T[eid]

  def assign_bcs(self, bc_list):
    return super().assign_bcs(bc_list)

  def assign_kernels(self, kernel_list):
    return super().assign_kernels(kernel_list)

class FVGradient:
  """
  Class to hold gradients for a FV object.
  Needs BoundaryCondition object since computing gradient will require information of the BC's
  """
  def __init__(self, m: mesh, volumeList: list, field: ScalarField, bc_list: list, tolerance=1e-6):
    self.bc_list = bc_list
    self.tolerance = tolerance
    self.volumeList = volumeList
    self.eids = []
    self.field = field # ScalarField that this gradient belongs to
    self.assign_elements() # fills up self.eids with element ids
    self.global_mesh = m
    self.gradient = VectorField(volumeList=self.volumeList, m=self.global_mesh, name=field.name+'_FVGradient')
    self.face_to_bc_idx = {} # key = face_id; value = bc_idx in bc_list
    self.get_bc_idxs() # fills face_to_bc_idx dictionary. -> now can do face as an index and get the bc out of it

    # initialize gradient
    self.updateGradient()

  def return_grad(self):
    return self.gradient.decompose_into_parts()


  def get_bc_idxs(self):
    for eid in self.eids:
      for fidx, is_boundary in enumerate(self.global_mesh.elements[eid].is_boundary):
        if is_boundary:
          # get face id
          face_id = self.global_mesh.elements[eid].face_ids[fidx]

          # check if face id in any of the boundaries in bc_list
          for bc_idx, bc in enumerate(self.bc_list):
            if face_id in bc.fid_list:
              self.face_to_bc_idx[face_id] = bc_idx
          if face_id not in self.face_to_bc_idx:
            raise Exception("Face id "+str(face_id)+" not in boundary conditions!")

  def get_face_gradient(self, eid: int, fid: int):
    """
    Interpolates gradient to face and returns it.
    Note: raises exception if the requested face is a boundary face.
    """
    # fid -> face id in faces[]

    # get element geo weighting factor
    face_index_C = self.global_mesh.elements[eid].face_ids.index(fid)
    gC = self.global_mesh.elements[eid].geoweights[face_index_C]
    gradC = self.gradient.get_value(eid)

    # neighbor F
    neighbor_id = self.global_mesh.elements[eid].neighbor_ids[face_index_C]
    face_index_F = self.global_mesh.elements[neighbor_id].face_ids.index(fid)
    gF = self.global_mesh.elements[neighbor_id].geoweights[face_index_F]
    gradF = self.gradient.get_value(neighbor_id)

    # evec
    evec = self.global_mesh.elements[eid].evec[face_index_C]

    # variables
    grad_phi_bar_f = gC*gradC + gF*gradF
    phiF = self.field.get_value(eid=neighbor_id)
    phiC = self.field.get_value(eid=eid)
    dCF = self.global_mesh.elements[eid].d_CNb[face_index_C]

    # Get grad face
    grad_face = grad_phi_bar_f + evec * (
      (phiF - phiC) / dCF - np.dot(grad_phi_bar_f, evec)
      )

    if self.global_mesh.elements[eid].is_boundary[face_index_C]:
      raise Exception("Cannot use this function on boundary faces.")

    return grad_face

  def assign_elements(self):
    for v in self.volumeList:
      for eid in v.elements:
        self.eids.append(eid)

  def updateGradient(self):
    """
    Updates gradient for the ScalarField
    """
    # Gradient calculation parts 1 and 2
    for _, eid in enumerate(self.eids):
      # Steps 1 and 2:
      grad_c = np.array([0.0, 0.0, 0.0])
      for fidx , face_id in enumerate(self.global_mesh.elements[eid].face_ids):

        # for faces that are internal faces.
        if self.global_mesh.elements[eid].is_face[fidx]:
          # Step 1. Calculate phi_f'
          neighbor_id = self.global_mesh.elements[eid].neighbor_ids[fidx]
          phi_fp = ( self.field.T[eid] + self.field.T[neighbor_id] ) / 2.0
          sv = self.global_mesh.faces[face_id].surface_vector

          # Step 2. Add to grad_c
          grad_c += sv * phi_fp
        else: # if it is a boundary face
          # get bc idx in self.bc_list
          bc_idx = self.face_to_bc_idx[face_id]
          this_bc_object = self.bc_list[bc_idx]
          grad_c += this_bc_object.get_gradient_contribution(face_id=face_id, eid=eid)

      # divide by volume to get grad_C
      grad_c = grad_c / self.global_mesh.elements[eid].volume

      # assign grad_C
      self.gradient.set_value(eid=eid, vec=grad_c)

    keep_going = True
    total_its = 0
    while keep_going & (total_its < 2):
      total_its += 1
      keep_going = False
      # Gradient calc part 3 and 4
      for _, eid in enumerate(self.eids):
        grad_c = np.array([0.0, 0.0, 0.0])
        for fidx , face_id in enumerate(self.global_mesh.elements[eid].face_ids):
          # If is an internal face
          if self.global_mesh.elements[eid].is_face[fidx]:
            # recompute phi_fp because i dont really want to store it anywhere and its fast anyways
            neighbor_id = self.global_mesh.elements[eid].neighbor_ids[fidx]
            phi_fp = ( self.field.T[eid] + self.field.T[neighbor_id] ) / 2.0

            # update phi_f
            first = (self.gradient.T[eid] + self.gradient.T[neighbor_id])
            second = self.global_mesh.faces[face_id].centroid  - 0.5 * (
              self.global_mesh.elements[eid].centroid + self.global_mesh.elements[neighbor_id].centroid )
            phi_f = phi_fp + 0.5 * np.dot(first, second)

            # assign phi_f to facefield from owner field
            self.field.faceField.assign_value(fid=face_id, val=phi_f)

            # now add to grad_c as step 4
            sv = self.global_mesh.faces[face_id].surface_vector
            grad_c += sv * phi_f
          else:
            # get bc idx in self.bc_list
            bc_idx = self.face_to_bc_idx[face_id]
            this_bc_object = self.bc_list[bc_idx]
            grad_c += this_bc_object.get_gradient_contribution(face_id=face_id, eid=eid)

        # divide by volume
        grad_c = grad_c / self.global_mesh.elements[eid].volume

        # compare values
        old_grad_values = self.gradient.get_value(eid)
        diff = np.array([0.0, 0.0, 0.0])

        # get relative differences
        for entry, _ in enumerate(diff):
          if grad_c[entry] == 0.0:
            diff[entry] = ( grad_c[entry] - old_grad_values[entry] ) # do abs diff if 0.0
          else:
            diff[entry] = np.abs(  ( grad_c[entry] - old_grad_values[entry]) / grad_c[entry]   )

        if any(diff >= self.tolerance):
          keep_going = True

        # assign grad_C
        self.gradient.set_value(eid=eid, vec=grad_c)

class FaceField:
  """
  A field that stores vaulues at the faces of elements.
  """
  def __init__(self, volumeList: list, m: mesh, name: str):
    self.volumeList = volumeList # a list of volume objects
    self.global_mesh = m
    self.name=name

class FaceScalarField(FaceField):
  def __init__(self, volumeList: list, m: mesh, name: str):
    super().__init__(volumeList=volumeList, m=m, name=name)
    self.T, self.fids = self.set_initial_condition()

  def set_initial_condition(self):
    _T = {}
    fids = []
    for v in self.volumeList:
      for fid in v.faces:
        _T[fid] = 0.0
        fids.append(fid)
    return _T, fids

  def assign_value(self, val: float, fid: int):
    self.T[fid] = val



