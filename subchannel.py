import subchannel # name of the so file.
import numpy as np

# make an array named x
x = np.array([1.0,2,3])

# runs subchannel on x
subchannel.hello_world(x)

# subchannel modifies the underlying values of what we passed in so the x arr should be changed in both python and cpp after running
print('python x after running is', x)

