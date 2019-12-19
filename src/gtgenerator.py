import keras
import tensorflow as tf
import numpy as np
import argparse
import os

from mymodel import model_inst
from util import V2VVoxelization
from dataicvl import ICVLHandDataset
from datamsra import MSRAHandDataset
from keras.utils import Sequence
from keras.models import load_model

dataset = 'MSRA'
test_subject_id = 1 #Change this to produce different ground truths
batch_size = 2 #Change

## Data, transform, dataset and loader
print('==> Preparing Data...')
if dataset == 'ICVL':
	keypoints_num = 16
	#data_dir = '..\\..\\Datasets\\ICVL Dataset'
elif dataset == 'MSRA':
	keypoints_num = 21
	test_subject_id = 1 #Change this to 3
	data_dir = '..\\..\\Datasets\\cvpr15_MSRAHandGestureDB'

center_dir = '..\\results\\centers\\' + dataset

#checkpoint_dir = '..\\results\\checkpoints'
checkpoint_dir = '..\\results\\checkpoints48\\model3\\checkpoints' #Change checkpoint directory in bucket for future jobs

print(data_dir)
print(center_dir)
print(checkpoint_dir)
print(os.path.exists(data_dir))
print(os.path.exists(center_dir))
print(os.path.exists(checkpoint_dir))
voxelization_train = V2VVoxelization(cubic_size = 200, augmentation = False) #Changed
voxelization_val = V2VVoxelization(cubic_size = 200, augmentation= False)

##Test
print('==> Testing...')
voxelize_input = voxelization_train.voxelize
evaluate_keypoints = voxelization_train.evaluate

def transform_val(sample):
	"""Data augmentation for validation data"""
	points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
	assert(keypoints.shape[0] == keypoints_num)
	inputs, heatmaps = voxelization_val({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
	return (inputs, keypoints)

def save_keypoints(filename, keypoints):
	#Reshape on sample keypoints into one line
	keypoints = keypoints.reshape(keypoints.shape[0], -1) #Each line should contain joint_num * 3 values
	np.savetxt(filename, keypoints, fmt='%0.4f')


#Dataset loader
if(dataset == 'ICVL'):
	val_set = ICVLHandDataset(data_dir, center_dir, 'test', transform_val)
	
elif(dataset == 'MSRA'):
	val_set = MSRAHandDataset(data_dir, center_dir, 'test', test_subject_id, transform_val)

keypoints = val_set._get_keypoints()
save_path = '..\\results\\test_s'+str(test_subject_id)+'_gt.txt' 
save_keypoints(save_path, keypoints)
