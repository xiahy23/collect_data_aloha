import h5py
import numpy as np

path = '/home/agilex/data/my_task/episode_5.hdf5'
with h5py.File(path, 'r') as f:
    action = f['/action'][:]
    qpos = f['/observations/qpos'][:]

print('shape:', action.shape)
print('min:', np.min(action))
print('max:', np.max(action))
print('mean:', np.mean(action))
print('std:', np.std(action))
print('has nan:', np.isnan(action).any())
print('max abs diff(action-qpos):', np.max(np.abs(action - qpos)))