class Field:
  def __init__(self, volumeList: list):
    self.volumeList = volumeList # a list of volume objects

class ZeroScalarField(Field):
  def __init__(self, volumeList: list):
    super().__init__(volumeList=volumeList)
    self.T, self.eids = self.assign_values()
    self.n = len(self.T) # size of field

  def assign_values(self):
    _T = {}
    eids = []
    for v in self.volumeList:
      for eid in v.elements:
        _T[eid] = 0.0
        eids.append(eid)
    return _T, eids


class ScalarField(Field):
  def __init__(self, initial_condition: Field):
    """
    volumeList: List of volume objects that this ScalarField lives on.
    """
    super().__init__(volumeList=initial_condition.volumeList)
    self.T = self.set_initial_condition(ic=initial_condition)
    self.eids = initial_condition.eids # copies element ids of the initial conditions

  def set_initial_condition(self, ic: Field):
    # we use the function to set the IC so that code creates a new copy -- self.T = ic.T would just pass a reference
    T = {}
    for key in ic.T.keys():
      T[key] = ic.T[key]
    return T

