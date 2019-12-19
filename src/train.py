"""Code for training the model built"""
import keras
import tensorflow as tf
import argparse
import os
import numpy as np

from src.mymodel_test import model_inst			#Changed
from src.util import V2VVoxelization		#Changed
from src.dataicvl import ICVLHandDataset	#Changed
from src.datamsra import MSRAHandDataset	#Changed
from keras.utils import Sequence
from keras.models import Model, load_model
from keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

## Some helpers
# def parse_args():
#     parser = argparse.ArgumentParser(description='Tensorflow Hand Keypoints Estimation Training')
#     #parser.add_argument('--resume', 'r', action='store_true', help='resume from checkpoint')
#     #parser.add_argument('--resume', '-r', default=-1, type=int, help='resume after epoch')
#     parser.add_argument('--dataset', '--d', default='MSRA', help='Dataset type')
#     parser.add_argument('--job-dir', '--j', default='results', help='MODEL_DIR')
#     args = parser.parse_args(['--dataset'])
#     return args 

dtype = tf.float32

#args = parse_args()
#resume_train = args.resume >= 0
#resume_after_epoch = args.resume

start_epoch = 0
epochs_num = 3

batch_size = 4

#Get dataset
#dataset = input('Enter dataset') 	
dataset = 'MSRA'
assert dataset in ('ICVL', 'MSRA'), 'Error: Dataset not recognized. Please enter "ICVL" or "MSRA"'

##Copying data to VM in GCP
if not os.path.exists(os.path.join(os.getcwd(), 'cvpr15_MSRAHandGestureDB')):
	current_dir = os.getcwd()
	os.system('gsutil -m cp -r gs://protean-atom-244816-engine/cvpr15_MSRAHandGestureDB ' + current_dir)
	os.system('gsutil -m cp -r gs://protean-atom-244816-engine/Research_Paper_Code/results ' + current_dir)
	# os.system('gsutil -m cp -r gs://protean-atom-244816-engine/checkpoints48/model3/checkpoints ' + current_dir) #Change checkpoint directory in bucket for future jobs.

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

save_checkpoint = True
#checkpoint_dir = '../results/checkpoints'
checkpoint_dir = os.path.join(current_dir,'results/checkpoints')
center_dir = os.path.join(current_dir,'results/centers/' + dataset)
#savedcheckpoint_dir = os.path.join(current_dir, 'checkpoints')
#center_dir = 'results\\centers\\' + dataset
print(os.path.abspath(data_dir))
print(os.path.abspath(center_dir))
print(os.path.abspath(checkpoint_dir))
#print(os.path.abspath(savedcheckpoint_dir))
print(os.path.exists(data_dir))
print(os.path.exists(center_dir))
print(os.path.exists(checkpoint_dir))
#print(os.path.exists(savedcheckpoint_dir))
 
cubic_size = 200

#Transform
voxelization_train = V2VVoxelization(cubic_size = 200, augmentation = True)  #Changed
voxelization_val = V2VVoxelization(cubic_size = 200, augmentation= False)

def transform_train(sample):
	"""Data augmentation for training data"""
	points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
	assert(keypoints.shape[0] == keypoints_num)
	inputs, heatmaps = voxelization_train({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
	return (inputs, keypoints)

def transform_val(sample):
	"""Data augmentation for validation data"""
	points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
	assert(keypoints.shape[0] == keypoints_num)
	inputs, heatmaps = voxelization_val({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
	return (inputs, keypoints)

#Dataset loader
if(dataset == 'ICVL'):
	# train_inputs = np.zeros(shape = (331006, 1, 88, 88, 88))
	# train_targets = np.zeros(shape = (331006, 21, 44, 44, 44))
	# for idx, i in enumerate(ICVLHandDataset(data_dir, center_dir, 'train', transform_train)):
	# 	train_inputs[idx] = i[0]
	# 	train_targets[idx] = i[1]
	train_set = ICVLHandDataset(data_dir, center_dir, 'train', transform_train)
	steps_per_epoch_train = 331006

	# val_inputs = np.zeros(shape = (1596, 1, 88, 88, 88))
	# val_targets = np.zeros(shape = (1596, 21, 44, 44, 44))
	#No separate validation dataset, just use test dataset instead
	# for idx, i in enumerate(ICVLHandDataset(data_dir, center_dir, 'test', transform_val)):
	# 	val_inputs[idx] = i[0]
	# 	val_targets[idx] = i[1]
	val_set = ICVLHandDataset(data_dir, center_dir, 'test', transform_val)
	steps_per_epoch_val = 1596
	
elif(dataset == 'MSRA'):
	# train_inputs = np.zeros(shape = (67893, 1, 88, 88, 88))
	# train_targets = np.zeros(shape = (67893, 21, 44, 44, 44))
	# for idx, i in enumerate(MSRAHandDataset(data_dir, center_dir, 'train', test_subject_id, transform_train)):
	# 	train_inputs[idx] = i[0]
	# 	train_targets[idx] = i[1]
	train_set = MSRAHandDataset(data_dir, center_dir, 'train', test_subject_id, transform_train)
	#steps_per_epoch_train = 67893
	#steps_per_epoch_train = 8486 #8
	steps_per_epoch_train = 16973 #4
	#steps_per_epoch_train = 22631 #3

	# val_inputs = np.zeros(shape = (8498, 1, 88, 88, 88))
	# val_targets = np.zeros(shape = (8498, 21, 44, 44, 44))
	#No separate validation dataset, just use test dataset instead
	# for idx, i in enumerate(MSRAHandDataset(data_dir, center_dir, 'test', test_subject_id, transform_val)):
	# 	val_inputs[idx] = i[0]
	# 	val_targets[idx] = i[1]
	val_set = MSRAHandDataset(data_dir, center_dir, 'test', test_subject_id, transform_val)
	#steps_per_epoch_val = 8498
	#steps_per_epoch_val = 1062 #8
	steps_per_epoch_val = 2124 #4
	#steps_per_epoch_val = 2832 #3

###############################################################################################################

##Model
print('==> Constructing Model...')
net = model_inst(input_channels = 1, output_channels = keypoints_num) 
adam = optimizers.Adam(lr=0.00025, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
net.compile(optimizer = adam, loss = 'mean_squared_error', metrics = ['accuracy'])
#net = load_model(os.path.join(savedcheckpoint_dir, 'model.h5'))
#print('==> Model Loaded Successfully!')
print('==> Model Built Successfully!')

################################################################################################################
#Resume
#If checkpoint directory is not empty
if len(os.listdir(checkpoint_dir)) > 2:
	checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir) #Change this to get the latest checkpoint (ie. 10)
	assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
	assert os.path.isfile(checkpoint_file), 'Error: no checkpoint file of epoch{}'.format(epoch)

	net.load_weights(checkpoint_file)
	print('Checkpoint weights loaded successfully into the model!')

################################################################################################################

#Train and evaluate
print('==> Training...')

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = os.path.join(checkpoint_dir,'cp-{epoch:04d}.ckpt')
cp_callback = ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every epoch.
    period=1)
net.save_weights(checkpoint_path.format(epoch=0))

# history = net.fit(train_inputs, train_targets, batch_size = batch_size, epochs = epochs_num, callbacks = [cp_callback], validation_data = (val_inputs, val_targets) )
history = net.fit_generator(train_set, steps_per_epoch = steps_per_epoch_train, epochs = epochs_num, verbose = 1, callbacks = [cp_callback], validation_data = val_set, validation_steps = steps_per_epoch_val, workers = 0, use_multiprocessing = False, shuffle = True, initial_epoch = 0) #Changed
print('\n==>History Dict:-', history.history)

#Saving the entire model
net.save(os.path.join(checkpoint_dir,'model.h5'))

#Copying checkpoints from VM to Bucket
os.system('gsutil cp -r ' + checkpoint_dir + ' gs://protean-atom-244816-engine/checkpoints49/model3') #Change checkpoint directory in bucket for future jobs
#results = net.evaluate(val_inputs, val_targets, batch_size = batch_size)
results = net.evaluate_generator(val_set, steps = steps_per_epoch_val, workers = 0, use_multiprocessing = False)
print('Test loss, Test accuracy: ', results)
print('Congratulations! Training and evaluation completed successfully!')

