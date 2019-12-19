"""Code for predicitons using the model built"""
import keras
import tensorflow as tf
import numpy as np
import argparse
import os

from src.mymodel import model_inst
from src.util import V2VVoxelization
from src.dataicvl import ICVLHandDataset
from src.datamsra import MSRAHandDataset
from keras.utils import Sequence
from keras.models import load_model

dataset = 'MSRA'
current_dir = os.getcwd()
batch_size = 4

##Copying data to VM in GCP
if not os.path.exists(os.path.join(os.getcwd(), 'cvpr15_MSRAHandGestureDB')):
	current_dir = os.getcwd()
	os.system('gsutil -m cp -r gs://protean-atom-244816-engine/cvpr15_MSRAHandGestureDB ' + current_dir)
	os.system('gsutil -m cp -r gs://protean-atom-244816-engine/Research_Paper_Code/results ' + current_dir)
	os.system('gsutil -m cp -r gs://protean-atom-244816-engine/checkpoints48 ' + current_dir) #Change checkpoint directory in bucket for future jobs


## Data, transform, dataset and loader
print('==> Preparing Data...')
if dataset == 'ICVL':
	keypoints_num = 16
	#data_dir = '..\\..\\Datasets\\ICVL Dataset'
	data_dir = 'gs://protean-atom-244816-engine/cvpr15_MSRAHandGestureDB'
elif dataset == 'MSRA':
	keypoints_num = 21
	test_subject_id = 3
	#data_dir = '..\\Datasets\\cvpr15_MSRAHandGestureDB'
	data_dir = os.path.join(current_dir, 'cvpr15_MSRAHandGestureDB')

center_dir = os.path.join(current_dir, 'results/centers/' + dataset)

#checkpoint_dir = '..\\results\\checkpoints'
checkpoint_dir = os.path.join(current_dir,'checkpoints48/model3/checkpoints') #Change checkpoint directory in bucket for future jobs

print(data_dir)
print(center_dir)
print(checkpoint_dir)
print(os.path.exists(data_dir))
print(os.path.exists(center_dir))
print(os.path.exists(checkpoint_dir))
voxelization_train = V2VVoxelization(cubic_size = 200, augmentation = False)
voxelization_val = V2VVoxelization(cubic_size = 200, augmentation= False)

##Test
print('==> Testing...')
voxelize_input = voxelization_train.voxelize
evaluate_keypoints = voxelization_train.evaluate


def transform_test(sample):
    points, refpoint = sample['points'], sample['refpoint']
    inputs = voxelize_input(points, refpoint)
    return inputs


# def transform_output(heatmaps, refpoints): #Needs to be changed or deleted if using keypoints
# 	refpoints = refpoints[:steps*batch_size]
# 	refjoints = refpoints.repeat(21, axis = 0)
# 	refjoints = refjoints.reshape((steps*batch_size, 21, 3))
# 	print('Refjoints shape = ',refjoints.shape)
# 	keypoints = evaluate_keypoints(heatmaps, refjoints)
# 	return keypoints

# class BatchResultCollector():
# 	def __init__(self, samples_num, transform_output):
# 		self.samples_num = samples_num
# 		self.transform_output = transform_output
# 		self.keypoints = None
# 		self.idx = 0

# 	def __call__(self, data_batch):
# 		inputs_batch, outputs_batch, refpoints_batch = data_batch
		
# 		keypoints_batch = self.transform_output(outputs_batch, refpoints_batch)

# 		if self.keypoints is None:
# 			#Initialize keypoints until dimensions available now
# 			self.keypoints = np.zeros((self.samples_num, *keypoints_batch.shape[1:]))

# 		batch_size = keypoints_batch.shape[0]
# 		self.keypoints[self.idx:self.idx + batch_size] = keypoints_batch
# 		self.idx += batch_size

# 	def get_result(self):
# 		return self.keypoints

def save_keypoints(filename, keypoints):
	#Reshape on sample keypoints into one line
	keypoints = keypoints.reshape(keypoints.shape[0], -1) #Each line should contain joint_num * 3 values
	np.savetxt(filename, keypoints, fmt='%0.4f')


if dataset == 'ICVL':
	test_set = ICVLHandDataset(data_dir, center_dir, 'test', transform_test)
	ref_set = test_set._get_ref_pts()
	steps = 1596

elif dataset == 'MSRA':
	test_set = MSRAHandDataset(data_dir, center_dir, 'test', test_subject_id, transform_test)
	ref_set = test_set._get_ref_pts()
	steps = 2124 #4
	#steps = 1000
#test_res_collector = BatchResultCollector(len(test_set)*batch_size, transform_output)

##Model
print('==> Loading Model...')
# net = model_inst(input_channels = 1, output_channels = keypoints_num)
# adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# net.compile(optimizer = adadelta, loss = 'mean_squared_error', metrics = ['accuracy'])
net = load_model(os.path.join(checkpoint_dir, 'model.h5'))
print('==> Model Loaded Successfully!')

print('==>Predicting on test set')

outputs = net.predict_generator(test_set, steps = steps, workers = 0, use_multiprocessing = False)
#test_res_collector((test_set, outputs, ref_set))
#keypoints_test = test_res_collector.get_result()
#save_keypoints('..\\results\\test_res.txt', keypoints_test)
save_keypoints(os.path.join(checkpoint_dir, 'test_res.txt'), outputs)

#Copying results from VM to Bucket
os.system('gsutil cp -r '+os.path.join(checkpoint_dir,'test_res.txt') + ' gs://protean-atom-244816-engine/checkpoints48') #Change checkpoint directory in bucket for future jobs

print('All done...')
