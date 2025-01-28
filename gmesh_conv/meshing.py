# imports

import gmsh
import numpy as np


# useful document for gmsh
# https://gmsh.info/doc/texinfo/gmsh.html


class face():
  # internal or external face with a boundary
  # boundary condition face - has a negative neighbor id
  def __init__(self, nodes: list, owner: int, neighbor: int, face_id: int):
    self.face_id = face_id
    self.nodes = nodes # node tags for each node on the face - presorted.
    self.node_coords = np.array([]) # coords for each node.
    self.own_id = owner # element eid of the owner
    self.neb_id = neighbor # element eid of the neighbor (-1 if this is external face)
    self.bndry = None # boundary that the face connects to (None of itnernal face)
    self.geo_center = None  # geometric center of surface.
    self.centroid = None # centroid of surface
    self.area = None
    self.surface_vector = None
    self.g = None # face weighting factor - see page 160
    self.set_coords()

  def set_area(self, area):
    self.area = area
  def set_surface_vector(self, surface_vector):
    self.surface_vector = surface_vector
  def set_bndry(self, bnd_id: int):
    self.bnd_id = bnd_id
  def set_geo_center(self, geometric_center: np.ndarray):
    self.geo_center = geometric_center
  def set_centroid(self, value):
    self.centroid = value
  def is_a_boundary_face(self):
    return (self.neb_id == -1)

  def set_coords(self):
    for node in self.nodes:
      cor, _1, _2, _3 = gmsh.model.mesh.getNode(node)
      try:
        self.node_coords = np.vstack([self.node_coords, cor])
      except:
        self.node_coords = cor

class element():
  def __init__(self, nodes: list, desc: str, ele_type: int, eid: int, sn: list, etag: int, entity_tag_dim: tuple):
    self.nodes = nodes # node tags for each node of this element
    self.sn = sn # surface node indexes for each node in this element.
    self.desc = desc # basic str describing thios element
    self.ele_type = ele_type # numeric type of this element
    self.eid = eid # mesh id of this elemeent associated with fv code - NOT the gmesh id
    self.num_surfs = len(sn) # number of surfaces on this element
    self.node_coords = np.array([])
    self.gmesh_element_tag = etag # gmesh element tag in the gmesh model
    self.entity_tag_dim = entity_tag_dim  # tuple (ent_tag, ent_dim) of the entity this element appears in
    self.entity_dims = [] # dimension of the entity each node appears on/in
    self.entity_tags = [] # tag of the entity each node appears on/in
    self.entity_tag_pairs = [] # tuples pairing the entity it appears on/in and dim
    self.surfaceNodeTags = [] # node tags for each surface [[node1,node2,node3, node4], [], [], ...]
    self.assign_surface_node_tags()
    self.get_node_info()

    # data for boundaries
    self.is_boundary = [False]*len(sn) # true or false if the surfaces of this element connect to bndry
    self.boundary_ids = [None]*len(sn) # boundary ids (none if not bndry but sid if it is)

    # data for internal faces
    self.is_face = [False]*len(sn) # bool whether or not it is an internal face.
    self.neighbor_ids = [None]*len(sn) # ids of the neighboring element -- element eid
    self.is_owner = [None]*len(sn) # int(1) if owner, int(-1) if neighbor for each face in sn

    # data for all faces
    self.face_ids = [None]*len(sn) # ids of the faces around this cell (including boundary and internal faces.)

    # some extra geometric info
    self.geo_center = None
    self.centroid = None
    self.volume = None
    self.geoweights = [None]*len(sn) # geometric weighting factor for each face
    self.evec = [None]*len(sn) # unit vectors to neighbors (Neighbor_centroid minus Centroid) Eq. 8.64
    self.d_CNb = [None]*len(sn) # distance to neighbor centroids (None if no neighbor)
    self.d_Cf = [None]*len(sn) # abs distance to faces (valid for internal faces and boundary faces)


  def set_centroid(self, T):
    self.centroid = T
  def set_geo_center(self, T):
    self.geo_center = T
  def set_volume(self, T):
    self.volume = T


  def verify_neighbors_and_faces(self):
    for idx, b in enumerate(self.is_boundary):
      if (b == True) & (self.is_face[idx] == True):
        raise Exception("eid =" + str(self.eid) + "A surface cannot be both a boundary and an internal face.")
      if (b == False) & (self.is_face[idx] == False):
        raise Exception("eid =" + str(self.eid) + "A surface must be either a boundary or a face.")

  def assign_neighbor_info(self, nodeTags: list, neighbor_id: int, face_id: int):
    for idx, this_node_tags in enumerate(self.surfaceNodeTags):
      sorted_this_node_tags = sorted(this_node_tags)
      sorted_input_node_tags = sorted(nodeTags)
      if sorted_this_node_tags == sorted_input_node_tags: # this is the correct surface.
        self.is_face[idx] = True
        self.face_ids[idx] = face_id
        self.neighbor_ids[idx] = neighbor_id

  def determine_if_boundary(self,surfaces):
    for idx, surfNodes in enumerate(self.surfaceNodeTags):
      # check if surfNodes is in any of surfaces.nodeTags
      for surf in surfaces:
        huge_set = set(surf.nodeTags)
        if all(item in huge_set for item in surfNodes):
          sid = surf.id
          self.boundary_ids[idx] = sid
          self.is_boundary[idx] = True

  def assign_surface_node_tags(self):
    # order = np.argsort(self.nodes)
    # self.nodes = sorted(self.nodes)
    # for surf_idx, surf in enumerate(self.sn):
    #   # surf is a vector of non-sorted nodes - e.g. [0,2,3]
    #   dummy = []
    #   for idx, i in enumerate(surf):
    #     self.sn[surf_idx][idx] = order[i]
    #     dummy += [self.nodes[order[i]]]
    #   self.surfaceNodeTags += [dummy]

    for surf_idx, surf in enumerate(self.sn):
      dummy = []
      for idx, i in enumerate(surf):
        dummy += [self.nodes[i]]
      self.surfaceNodeTags += [dummy]

  def get_node_info(self):
    self.entity_dims = []
    self.entity_tags = []
    for node in self.nodes:
      coords, parcoords, e_dim, e_tag = gmsh.model.mesh.getNode(node)
      try:
        self.node_coords = np.vstack([self.node_coords, coords])
      except:
        self.node_coords = coords
      self.entity_dims += [e_dim]
      self.entity_tags += [e_tag]
      self.entity_tag_pairs += [(e_dim, e_tag)]

class surface():
  def __init__(self, dim, etag, name, ptag, id: int):
    self.id = id # surface id used to refer to surfce internally
    self.dim = dim
    self.name = name
    self.etag = etag # entity tag that this surface belongs to
    self.ptag=ptag # physical tag this surface belongs to
    self.physicalName = name
    self.elements = []
    self.nodeTags, self.coords = gmsh.model.mesh.getNodesForPhysicalGroup(self.dim, self.ptag)
    self.face_ids = [] # face ids that belong to this surface
  def assign_element(self, element):
    self.elements += [element]
  def assign_face_ids(self, face_id: int):
    self.face_ids += [face_id]


class volume(surface):
  def __init__(self, dim, etag, name, ptag, id):
    super().__init__(dim, etag, name, ptag, id)
    self.elements = []
    self.faces = []
  def append_element(self, eid: int):
    self.elements.append(eid)
  def append_face(self, fid):
    self.faces.append(fid)

class mesh():
  def __init__(self, mesh_id: int, elements: list, boundaries: list, volumes: list, faces: list, orthogonalityApproach: str):
    self.orthogonalityApproach = orthogonalityApproach
    self.boundaries = boundaries
    self.mesh_id = mesh_id
    self.elements = elements
    self.volumes = volumes
    self.faces = faces

    # orthogonal, nonorthogonal, and surface vector components for every mesh element
    # dict[element_id / eid][list of vectors]
    self.Ef = {}
    self.Tf = {}
    self.gDiffs = {} # dict[eid][surface_idx] geoemtric diffusion coeffs -- diff defs for faces and boundaries.
    self.calculateNonorthogonalComponents()

  def calculateNonorthogonalComponents(self):
    # Decomposes Sf into Sf = Ef + Tf
    # This method is how Ef and Tf are computed.

    for this_element in self.elements:
      eid = this_element.eid
      this_Ef = []
      this_Tf = []
      this_gDiff = [] # geometric diffusion coeffs

      e = self.elements[eid] # element with eid
      evecs = e.evec # list of normal vectors for each surface

      # now for every surface calculate Ef Tf and Sf
      for idx, faceid in enumerate(e.face_ids):
        if e.is_owner[idx]: #
          multiplier = 1.0 # S poiints away from cell for owner cells
        else:
          multiplier = -1.0
        Sf = self.faces[faceid].surface_vector * multiplier
        area = self.faces[faceid].area
        evec = evecs[idx]

        if evec is not None: # this should pass no matter what - exception raised below just in case
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
            this_gDiff.append( np.linalg.norm(_ef) / e.d_CNb[idx])
          elif e.is_boundary[idx]: # for boundaries
            this_gDiff.append(  np.linalg.norm(_ef) / e.d_Cf[idx] )
          else:
            raise Exception("Neither a face or boundary, huh????")

        else: # if it is a boundary face
          raise Exception("evec should be assigned for all faces and boundaries.")

      # now that we iterated over all faces, add to dictionaries
      self.Ef[eid] = this_Ef
      self.Tf[eid] = this_Tf
      self.gDiffs[eid] = this_gDiff


  def export_to_vtk(self, filename, fieldlist=[]):
    # Takes a mesh of points and exports to VTK format
    node_dict = {}
    i = int(0)
    total_celllist_len = int(0)

    for e in self.elements:
      total_celllist_len += 1 # add 1 here - not a mistake
      for node_coord in e.node_coords:
        total_celllist_len += 1 # add here as well - not a mistake
        if tuple(node_coord) not in node_dict:
          node_dict[tuple(node_coord)] = i
          i += 1

    with open(filename, "w") as vtk_file:
      """
      Writes the given list of Point objects to a .vtk file for visualization in ParaView.

      Args:
          points (list): List of Point objects.
          filename (str): Name of the output .vtk file.
      """
      # Write header
      vtk_file.write("# vtk DataFile Version 2.0\n")
      vtk_file.write("VTK file generated from Point objects\n")
      vtk_file.write("ASCII\n")
      vtk_file.write("DATASET UNSTRUCTURED_GRID\n")

      # write poitns now where points are the vertexes of each cell
      vtk_file.write(f"POINTS {len(node_dict)} float\n")
      for node_key in node_dict.keys():
        vertex = list(node_key)[0] #
        vtk_file.write(f"{list(node_key)[0]} {list(node_key)[1]} {list(node_key)[2]} \n")

      # cells now - CELLS n size - n=number cells - size = numbercells*numNodes + numCells
      vtk_file.write(f"CELLS {len(self.elements)} {total_celllist_len}\n")
      for i, cell in enumerate(self.elements):
        line = str(len(cell.node_coords))
        for coord in cell.node_coords:
          line += " " + str(node_dict[tuple(coord)])

        vtk_file.write(line+"\n")

      # cell_types now --- CELL_TYPES n where n = number of cells
      vtk_file.write(f"CELL_TYPES {len(self.elements)}\n")
      for cell in self.elements:
        if cell.ele_type == 4: # 4 node tetrahedron -> vtk type 10
          vtk_file.write(f"10\n")
        elif cell.ele_type == 7: # gmesh 5 node pyramid -> vtk type 14
          vtk_file.write(f"14\n")

      # cell data and field data now
      # ----------------------------
      # CELL_DATA numCells
      # FIELD FieldData numFields
      # fieldname1 1 numCells dataType (double or float)
      # values.....
      # values.....
      #
      # fieldname2 1 numCells dataType (double or float)
      # values.....
      # values.....
      #


      vtk_file.write(f"CELL_DATA {len(self.elements)}\n")
      vtk_file.write(f"FIELD FieldData "+str(1+len(fieldlist))+" \n")
      vtk_file.write(f"eid 1 {len(self.elements)} double\n")
      line = ""
      counter = 0
      for i, cell in enumerate(self.elements):
        line = line + str(cell.eid) + " "
        if counter == 10:
          vtk_file.write(f"{line} \n")
          line = ""
          counter = 0
        counter += 1
      vtk_file.write(f"{line} \n")

      for field_idx, field in enumerate(fieldlist):
        if field.name is None:
          raise Exception("Field "+str(field_idx)+" has no fieldName attribute")
        line = ""
        counter = 0
        vtk_file.write(f"{field.name} 1 {len(self.elements)} double\n")
        for i, cell in enumerate(self.elements):
          if cell.eid in field.eids:
            line = line + str(fieldlist[field_idx].T[i]) + " "
          else:
            line + str(0.0) + " "
          if counter == 10:
            vtk_file.write(f"{line} \n")
            line = ""
            counter = 0
          counter += 1
        vtk_file.write(f"{line} \n")



def calculate_triangle_area(coords: np.ndarray):
  """
  # Gets area of a triangle ----
  coords = np.array([ [x1,y1,z1],
                      [x2,y2,z2],
                      [x3,y3,z3] ])
  """
  # gets area of triangle
  if len(coords) != 3:
    print(coords)
    raise Exception("Length of triangle coords must be 3!")
  A = coords[0]
  B = coords[1]
  C = coords[2]
  AB = B - A
  AC = C - A
  surface_vector = np.cross(AB,AC) * 1.0 / 2.0
  # magnitude = np.linalg.norm(cross_product)
  # area = 1.0/2.0 * magnitude
  area = np.linalg.norm(surface_vector)
  return area, surface_vector


def get_element_volume_info(e, faces):
  # calculate volumes, geo center and centroid of each finite volume element.
  # we do this by dividing ALL polyhedron into smaller pyramids.
  # then we get the volumes of all subpyramids and use them to calculate centroid of larger polyhedra

  # basic info to work with for this element
  coords = e.node_coords
  fids = e.face_ids # face ids

  # first get geometric center
  avgXYZ = np.mean(coords, axis=0)

  # now calculate volumes and geo centers

  # iterate across all face ids to form subpyramids
  subpyramid_volumes = np.array([])
  subpyramid_centroidsX = np.array([])
  subpyramid_centroidsY = np.array([])
  subpyramid_centroidsZ = np.array([])
  for fid in fids:
    # info needed for the face.
    face_area = faces[fid].area
    face_centroid = faces[fid].centroid
    face_surf_vec = faces[fid].surface_vector


    # get centroid of the pyramid
    centroid_subpyramid = 0.75 * face_centroid + 0.25 * avgXYZ

    # get distance from geometric center to surface centroid
    dGf = face_centroid - avgXYZ

    norm = face_surf_vec / np.linalg.norm(face_surf_vec)

    # get volume of subpyramid
    volume_subpyramid = face_area * abs(np.dot(dGf, norm)) / 3.0

    # append information
    subpyramid_volumes = np.append(subpyramid_volumes, volume_subpyramid)
    subpyramid_centroidsX = np.append(subpyramid_centroidsX, centroid_subpyramid[0])
    subpyramid_centroidsY = np.append(subpyramid_centroidsY, centroid_subpyramid[1])
    subpyramid_centroidsZ = np.append(subpyramid_centroidsZ, centroid_subpyramid[2])

  # now get centroid and volume of the whole thing
  tot_vol = np.sum(subpyramid_volumes)
  polyhedron_centroidX = np.sum(subpyramid_centroidsX * subpyramid_volumes) / tot_vol
  polyhedron_centroidY = np.sum(subpyramid_centroidsY * subpyramid_volumes) / tot_vol
  polyhedron_centroidZ = np.sum(subpyramid_centroidsZ * subpyramid_volumes) / tot_vol

  volume_centroid = np.array([polyhedron_centroidX, polyhedron_centroidY, polyhedron_centroidZ])

  return tot_vol, volume_centroid, avgXYZ

def mesh_from_gmsh(filename: str, orthogonalityApproach: str):
  # Initialize gmsh
  gmsh.initialize()
  gmsh.open(filename)
  entities = gmsh.model.getEntities() # vector of [dim, tag] pairs for each entity - points,surfaces,vols. tags are the ids of the entity - e.g. point 1, surface 1, volume 1.
                                      # each dimension has a new set of tags - e.g. you can have point 1 and surface 1 since surf and point are in different dimensions

  # surfaces (boundaries of volumes)
  surfaces = []
  sid = 0

  # volumes (volumes in the gmsh)
  volumes = []
  vid = 0
  volume_dict = {} # keys are a tuple for this entity: (ent_tag, ent_dim). values are ids of the volume

  # get all entities in the mesh
  for e in entities:
    dim = e[0]
    tag = e[1]
    # Get the mesh nodes for the entity (dim, tag):
    nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(dim, tag)

    # Get the upward and downward adjacencies of the entity.
    up, down = gmsh.model.getAdjacencies(dim, tag)

    # Get physical tags
    physicalTags = gmsh.model.getPhysicalGroupsForEntity(dim, tag)

    # determine if this is a labelled surface or possibly a boundary
    if dim == 2: # surface
      if len(physicalTags) >= 2:
        raise Exception("Surface cannot belong to more than 1 physical group.")
      elif len(physicalTags == 1):
        # is a surface with a name - e.g boundary
        name = gmsh.model.getPhysicalName(dim, physicalTags[0])
        this_surf = surface(dim=dim, etag=tag, name=name, ptag=physicalTags[0], id=sid)
        sid += 1
        surfaces += [this_surf]

    # determine if this is a volume
    if dim == 3:
      if len(physicalTags) >= 2:
        raise Exception("Volume cannot belong to more than 1 physical group.")
      elif len(physicalTags == 1):
        # is a volume with a name
        name = gmsh.model.getPhysicalName(dim, physicalTags[0])
        this_volume = volume(dim=dim, etag=tag, name=name, ptag=physicalTags[0], id=vid)

        # append to volumes list and dict
        volumes += [this_volume]
        volume_dict[(tag, 3)] = vid
        vid += 1

  ################################
  # PARSING ELEMENTS
  ################################

  # getting all elements in the mesh and adding some basic data
  elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(-1, -1) # gets all elements in the mesh.
  # see pdf page 367 - https://gmsh.info/dev/doc/texinfo/gmsh.pdf
  # element type 15 = a point (dont care)
  # element type 1 = 2 node line (useful for boundaries?)
  # element type 2 = 3 node triangle (useless)
  # element type 3 = 4 node quadrangle (2d aka garbage)
  # element type 4 = 4 node tetrahedon (3d keep)
  # element type 7 = 5 node pyramid (3d keep)
  elements = []
  element_id = 0
  for idx, t in enumerate(elementTypes):
    nt = nodeTags[idx] # node tags
    et = elementTags[idx] # element tags

    if t in [15,1,3,2]:
      _ = 1 # do nothing since these are garbage

    elif t == 4: # 4 node tetrahedons
      numNodes = 4
      for nt_idx, this_node_tag in enumerate(nt):
        if nt_idx % numNodes == 0:
          # get node indexes for this element
          n0 = nt[nt_idx]
          n1 = nt[nt_idx+1]
          n2 = nt[nt_idx+2]
          n3 = nt[nt_idx+3]
          sn0 = [0, 2, 3] # nodes that make up each surface of this elements
          sn1 = [0, 1, 2]
          sn2 = [1, 2, 3]
          sn3 = [0, 1, 3]
          sn = [sn0, sn1, sn2, sn3]

          # get some specific element data
          _, _, entity_dim, entity_tag = gmsh.model.mesh.getElement(et[int((nt_idx+1)/numNodes)]) # tag and dim of the entity this element appears on
          ent_tag_dim = (entity_tag, entity_dim)

          elements += [element(nodes=[n0, n1, n2, n3], ele_type=4, desc='4n_tet', eid=element_id, sn=sn, entity_tag_dim=ent_tag_dim, etag=et[int((nt_idx+1)/numNodes)])]
          element_id += 1

    elif t == 7: # 5 node pyramid
      numNodes = 5
      for nt_idx, this_node_tag in enumerate(nt):
        if nt_idx % numNodes == 0:
          # get node indexes for this element
          n0 = nt[nt_idx]
          n1 = nt[nt_idx+1]
          n2 = nt[nt_idx+2]
          n3 = nt[nt_idx+3]
          n4 = nt[nt_idx+4]
          sn0 = [0,1,2,3]
          sn1 = [1,2,4]
          sn2 = [2,3,4]
          sn3 = [0,3,4]
          sn4 = [0,1,4]
          sn = [sn0, sn1, sn2, sn3, sn4]

          # get some specific element data
          _, _, entity_dim, entity_tag = gmsh.model.mesh.getElement(et[int((nt_idx+1)/numNodes)]) # tag and dim of the entity this element appears on
          ent_tag_dim = (entity_tag, entity_dim)

          elements += [element(nodes=[n0, n1, n2, n3, n4], ele_type=7, desc='5n_pyr', eid=element_id, sn=sn, entity_tag_dim=ent_tag_dim, etag=et[int((nt_idx+1)/numNodes)])]
          element_id += 1

    else:
      print("element type  =", t)
      raise Exception("invalid element type = "+str(t))


  # Determining if each of the elements has a face on the boundary using elements and my boundary surfaces
  for e in elements:
    e.determine_if_boundary(surfaces)

  ################################
  # PARSING ELEMENTS NEIGHBORS
  ################################

  # Getting the neighbors for all the elements in the mesh
  internal_face_map = {}
  sorted_to_unsorted_dict = {} # key is sorted node tags and value is the unsorted ones
  for idx, e in enumerate(elements):
    for idx_this_elements_surf, surface_i in enumerate(e.surfaceNodeTags):
      if (e.is_boundary[idx_this_elements_surf] == False):
        surface_key = tuple(sorted(surface_i))
        unsorted_key = tuple(surface_i)
        if (surface_key not in internal_face_map):
          internal_face_map[surface_key] = []
          sorted_to_unsorted_dict[surface_key] = unsorted_key
        internal_face_map[surface_key].append(idx)

  # # internal_face_map has tuples that are the keys to the dict where tuples are the nodeTags that make up the surface.
  # internal_face_map values are [parent, neighbor] for the face.

  ################################
  # making faces for pairs
  ################################

  # now assign neighbors for each surface in element
  faces = []
  face_id = 0
  for nodeTags in internal_face_map.keys():
    owner_neighbor = internal_face_map[nodeTags]
    owner_id = owner_neighbor[0]
    neighbor_id = owner_neighbor[1]
    unsorted_node_tags = sorted_to_unsorted_dict[nodeTags]
    faces += [face(nodes=unsorted_node_tags, owner=owner_id, neighbor=neighbor_id, face_id=face_id)]

    elements[owner_id].assign_neighbor_info(nodeTags=nodeTags, neighbor_id=neighbor_id, face_id=face_id)
    elements[neighbor_id].assign_neighbor_info(nodeTags=nodeTags, neighbor_id=owner_id, face_id=face_id)
    face_id += 1

  # now make faces for the face-boundaries - neighbor id are the negative of the boundary(surface) ids
  for e in elements:
    for idx, b in enumerate(e.is_boundary):
      if b: # if True
        bnd_id = int(e.boundary_ids[idx])
        nodeTags = e.surfaceNodeTags[idx]
        faces += [face(nodes=nodeTags, owner=e.eid, neighbor=-1, face_id = face_id)]
        faces[face_id].set_bndry(bnd_id)
        e.face_ids[idx] = face_id
        face_id += 1
        # print(face_id-1, nodeTags)


  # verify elements that all faces are either a boundary or a face
  for e in  elements:
    e.verify_neighbors_and_faces()

  ################################
  # FACE AREAS AND CENTROIDS
  ################################

  # compute area and centroids and geometric centers of all faces.
  for f in faces:
    numNodes = len(f.nodes)
    nodes = []
    for n in f.nodes:
      coords = f.node_coords
      if len(coords) == 3:
        # already a triangle
        avgXYZ = np.mean(coords, axis=0)
        f.geo_center = avgXYZ # assigns geoemetric center for this triangle.
        f.centroid = f.geo_center
        area, surface_vector = calculate_triangle_area(coords)
        f.set_area(area)

        # normalization to 1.0 and multiply by area of triangle. see larger comment in else: statement below.
        subtriangle_area = np.linalg.norm(surface_vector)
        surface_vector = surface_vector / subtriangle_area * area
        f.set_surface_vector(surface_vector)

      else:
        if f.face_id == 3069:
          asdasdasd = 1
        # else it is a shape with more than 3 nodes and we need to cut it into smaller triangles.
        area = 0.0
        sub_triangle_x = np.array([])
        sub_triangle_y = np.array([])
        sub_triangle_z = np.array([])
        sub_triangle_areas = np.array([])

        # get geo center
        avgXYZ = np.mean(coords, axis=0)

        # iterating over subtriangles of the >3 node polygon
        for i, coord in enumerate(coords):
          ### for this function the coords must be in a circle - e.g. adjacent nodes for adjacent indexes.
          ### must also be a convex shape i think?

          if i == (len(coords)-1): # last coord
            rightCoord = coords[0]
          else:
            rightCoord = coords[i+1]

          # get geo center

          triangleCoords = coord
          triangleCoords = np.vstack([triangleCoords, rightCoord])
          triangleCoords = np.vstack([triangleCoords, avgXYZ])


          this_area, surface_vector = calculate_triangle_area(triangleCoords)
          area += this_area
          sub_triangle_areas = np.append(sub_triangle_areas, this_area)

          this_triangle_centroid = np.mean(triangleCoords, axis=0) # geometric mean = centroid for triangle

          sub_triangle_x = np.append(sub_triangle_x, this_triangle_centroid[0])
          sub_triangle_y = np.append(sub_triangle_y, this_triangle_centroid[1])
          sub_triangle_z = np.append(sub_triangle_z, this_triangle_centroid[2])

        # do a weighted sum of all subtriangles centroids and their areas to get the face centroid
        polygonCentroidX = sum(sub_triangle_x * sub_triangle_areas) / area
        polygonCentroidY = sum(sub_triangle_y * sub_triangle_areas) / area
        polygonCentroidZ = sum(sub_triangle_z * sub_triangle_areas) / area

        f.set_centroid(np.array([polygonCentroidX, polygonCentroidY, polygonCentroidZ]))

        # set geo center

        f.set_geo_center(avgXYZ)

        # set area
        f.set_area(area)

        # normalize surface vector of subtriangle to 1.0, then normalize it to area of the whole surface.
        # the surface vector of the subtriangle will be in same direction as the surf vector of the surface
        # so we just need to scale it according to area.
        subtriangle_area = np.linalg.norm(surface_vector)
        surface_vector = surface_vector / subtriangle_area * area
        f.set_surface_vector(surface_vector)

  ################################
  # ELEMENT VOLUMES AND CENTROIDS
  ################################

  for e in elements:
    tot_vol, volume_centroid, avgXYZ = get_element_volume_info(e, faces)
    # assign stuff
    e.set_geo_center(avgXYZ)
    e.set_volume(tot_vol)
    e.set_centroid(volume_centroid)

  # now get total volume and print it
  volumeTOT = 0.0
  for e in elements:
    volumeTOT += e.volume
  print('============================')
  print('Total volume is:', volumeTOT)


  ################################
  # GEOMETRIC WEIGHTING FACTORS
  ################################

  # now make geometric weighting factors for each face in an element
  for e in elements:
    for idx, fid in enumerate(e.face_ids):
      if not e.is_boundary[idx]:
        # only get geo weighting factor if it is NOT a boundary

        # get distance from E_centroid to face centroid
        dcf = e.centroid - faces[fid].centroid

        # get and normalize surface vector
        sv = faces[fid].surface_vector
        ef = sv / np.linalg.norm(sv) # normalize to 1.0

        # now get neighbor distance from neightbor centroid to face centroid
        nid = e.neighbor_ids[idx]
        dff = elements[nid].centroid - faces[fid].centroid

        # now get geo weighting factor

        dotE = np.abs( np.dot(dcf, ef) )
        dotF = np.abs( np.dot(dff, ef) )
        geoweight = dotE/(dotE + dotF)
        e.geoweights[idx] = geoweight

  ################################
  # Calculate unit distance vector to neighbor centroids
  ################################
  for e in elements:
    for idx, is_f in enumerate(e.is_face):
      if is_f: # if it is a boundary

        # get neighbor id
        nid = e.neighbor_ids[idx]

        # get vector from Centroid to neighbor
        dvec = elements[nid].centroid - e.centroid
        d = abs(np.linalg.norm(dvec)) # absolute distance

        # assign abs distance and unit distance
        e.d_CNb[idx] = d # assign distance to neighbors
        e.evec[idx] = dvec / d # unit distance vector
      else:
        # for boundaries e is from centroid to centroid of face
        fid = e.face_ids[idx]
        dvec = faces[fid].centroid - e.centroid
        d = abs(np.linalg.norm(dvec)) # absolute distance
        e.evec[idx] = dvec / d # unit distance vector centroid to face centroid

  ################################
  # Calculate abs distance to faces bounding this element
  ################################
  for e in elements:
    for idx, f in enumerate(e.face_ids):
      distance_vector = faces[f].centroid - e.centroid

      # assign abs distance to face
      e.d_Cf[idx] = abs(np.linalg.norm(distance_vector))


  ################################
  # Reorient surface vectors of each face.
  ################################
  # face surface vectors should point AWAY from the owner cell and TOWARDS the neighbor cell
  for f in faces:
    this_surface_vector = f.surface_vector

    if any(this_surface_vector == None):
      raise Exception("surface vector for face_id="+str(f.face_id)+" was never assigned")

    if f.face_id == 9214:
      aasdasdasdsad = 1
    this_centroid = f.centroid
    owner_centroid = elements[f.own_id].centroid

    d = this_centroid - owner_centroid

    dotProd = np.dot(d, this_surface_vector)

    if dotProd < 0: # oriented wrong way so we flip it
      f.surface_vector = f.surface_vector * -1.0
    elif dotProd == 0:
      raise Exception("Somehow the face surface vector and the vector d are perpendicular?")

  ################################
  # Tell each element if it is owner or neighbor of its faces
  ################################
  for e in elements:
    for idx, fid in enumerate(e.face_ids):
      if faces[fid].own_id == e.eid:
        e.is_owner[idx] = True
      elif faces[fid].neb_id == e.eid:
        e.is_owner[idx] = False
      else:
        raise Exception("Not sure if this element is the owner or the neighbor of this face.")

  ################################
  # For each surface make a list of the faces that it has.
  ################################
  for ele in elements:
    for fidx, fid in enumerate(ele.face_ids):
      if ele.is_boundary[fidx]:
        bndry_id = faces[fid].bnd_id
        surfaces[bndry_id].assign_face_ids(face_id = fid)

  # now get total surface area of the mesh
  tot_surf_area = 0.0
  for s in surfaces:
    this_b_sv = np.array([0.0,0.0,0.0])
    for face_id in s.face_ids:
      tot_surf_area += faces[face_id].area
      this_b_sv += faces[face_id].surface_vector
    print("Surface vector for surface", s.name, "is", this_b_sv)
  print('Total bndry surface area is:', tot_surf_area)
  print('============================')

  ################################
  # Assign elements to volumes
  ################################
  for e in elements:
    volume_key = e.entity_tag_dim
    volume_id = volume_dict[volume_key]
    volumes[volume_id].append_element(eid=e.eid)

  ################################
  # Assign faces to volumes
  ################################
  temp_face_list = []
  for e in elements:
    volume_key = e.entity_tag_dim
    for fid in e.face_ids:
      if fid not in temp_face_list:
        temp_face_list.append(fid)
        volumes[volume_id].append_face(fid=fid)



  ################################
  # MAKING MESH OBJECT
  ################################
  m = mesh(mesh_id=0, boundaries=surfaces, volumes=volumes,
           elements=elements, faces=faces, orthogonalityApproach=orthogonalityApproach)

  ################################
  # Print mesh information
  ################################
  print("Information for mesh with id of", m.mesh_id)
  print("Volume information:")
  for v in m.volumes:
    print("\tVolume ID "+str(v.id)+"  Physical Name: "+str(v.name))

  print("Surface information:")
  for v in m.boundaries:
    print("\tSurface ID "+str(v.id)+"  Physical Name: "+str(v.name))

  return m

