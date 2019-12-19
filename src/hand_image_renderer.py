import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datamsra import load_depthmap
import os

from datamsra import MSRAHandDataset
from util import V2VVoxelization

img_width = 320
img_height = 240
min_depth = 100
max_depth = 700

## Depth Image Renderer
dataset = 'MSRA'

database_dir  = '..\\..\\Datasets\\cvpr15_MSRAHandGestureDB'
center_dir = '..\\results\\centers\\' + dataset
filename = os.path.join(database_dir, 'P0\\1\\000000_depth.bin')
A = load_depthmap(filename, img_width, img_height, max_depth)
plt.imshow(A, cmap='gray')
plt.show()

## Voxelized Image Renderer

keypoints_num = 21
test_subject_id = 3
data_dir = '..\\..\\Datasets\\cvpr15_MSRAHandGestureDB'

#Transform
voxelization_train = V2VVoxelization(cubic_size = 200, augmentation = False)
voxelization_val = V2VVoxelization(cubic_size = 200, augmentation= False)

def transform_train(sample):
    """Data augmentation for training data"""
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)
    inputs, heatmaps = voxelization_train({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
    return (inputs, keypoints)

train_set = MSRAHandDataset(data_dir, center_dir, 'train', test_subject_id, transform_train)

hand3d = np.zeros(shape=(88, 88, 88))
hand3d = train_set[0][0][0][0]

B = np.where(hand3d == 1)
X, Y, Z = B
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_trisurf(X, Y, Z)
plt.show()

