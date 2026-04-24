import h5py
import numpy as np

path = '/home/agilex/data/my_task/episode_0.hdf5'
with h5py.File(path, 'r') as f:
    action = f['/action'][:]

delta = np.abs(np.diff(action, axis=0))
print('left max delta:', np.max(delta[:, :7]))
print('right max delta:', np.max(delta[:, 7:]))
print('left mean delta:', np.mean(delta[:, :7]))
print('right mean delta:', np.mean(delta[:, 7:]))