from meta_pointnet_transfer_attention import MetaLearner
from data_utils.S3DISDataLoader import S3DISDatasetFewShot,ScannetDatasetFewShot
from models.pointnet2_sem_seg_msg_attention import get_model

import argparse
import os
import torch
from torch.autograd import Variable
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
	parser = argparse.ArgumentParser('Model')
	parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
	parser.add_argument('--epoch', default=500, type=int, help='Epoch to run [default: 500]')
	parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
	parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
	
	parser.add_argument('--n_way', type=int, default=4, help='n_way category')
	parser.add_argument('--k_shot', type=int, default=2, help='k_way sample')
	# parser.add_argument('--k_query', type=int, default=2, help='k_query sample')
	# parser.add_argument('--base_lr', default=1e-3, type=float, help='內循環(support set) >> 學習率(α:base lr) = inner_lr')
	parser.add_argument('--meta_lr', default=1e-3, type=float, help='外循環(query set) >> 學習率(β:meta lr) = outer_lr') 
	parser.add_argument('--meta_batchsz', type=int, default=8, help='Meta batch Size during meta training [default: 8]')
	parser.add_argument('--num_updates', default=5, type=int, help='number of reptile update in one task')
	parser.add_argument('--type', type=str, default='NONE', help='attention type name [default: NONE >> mean no load pretain and no attention / noAttention:mean load pretrain and no attention][scSE,ECA_scSE,ECANet,SENet,CBAM,noAttention,NONE]')
	parser.add_argument('--dataset', type=str, default='S3DIS', help='dataset [default: S3DIS][S3DIS,PersonS3DIS,Scannet]')
	
	return parser.parse_args()

# 預處理點
# def preprocess_point(points,labels,mode):
# 	points = points.data.numpy()
# 	# 訓練時才做數據增強
# 	if (mode == 'train'):
# 		# 旋轉具有法向量信息的點雲做數據增強
# 		points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
# 	points = torch.Tensor(points)
# 	points, labels = points.float().cuda(), labels.long().cuda()
# 	points = points.transpose(2, 1)
# 	return points, labels

# 預處理點
def preprocess_point(points,labels):
	points = points.data.numpy()
	# 旋轉具有法向量信息的點雲做數據增強
	points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
	points = torch.Tensor(points)
	points, labels = points.float().cuda(), labels.long().cuda()
	points = points.transpose(2, 1)
	return points, labels

def get_class2Label(class_txt_path):
	# class_txt_path='datasets/class_names.txt'
	with open(class_txt_path,'r') as file:
		line = file.read()
	class2Label_dict={index: value for index, value in enumerate(line.splitlines())}
	return class2Label_dict
def prtrain_model(attention_type):
	if(attention_type=='ECANet'):
		pretrain_state_dict = torch.load('/media/g111056119/SP PHD U3/datasets/log/pointnet/s3dis/ranger/attention/ECANet/7batches_4ways_2shots_500epchoes/checkpoints/best_model.pth')
	elif(attention_type=='SENet'):
		pretrain_state_dict = torch.load('/media/g111056119/SP PHD U3/datasets/log/pointnet/s3dis/ranger/attention/SENet/7batches_4ways_2shots_500epchoes/checkpoints/best_model.pth')
	elif(attention_type=='scSE'):
		pretrain_state_dict = torch.load('/media/g111056119/SP PHD U3/datasets/log/pointnet/s3dis/ranger/attention/scSE/7batches_4ways_2shots_500epchoes/checkpoints/best_model.pth')
	elif(attention_type=='ECA_scSE'):
		pretrain_state_dict = torch.load('/media/g111056119/SP PHD U3/datasets/log/pointnet/s3dis/ranger/attention/ECA_scSE/7batches_4ways_2shots_500epchoes/checkpoints/best_model.pth')
	elif(attention_type=='CBAM'):
		pretrain_state_dict = torch.load('/media/g111056119/SP PHD U3/datasets/log/pointnet/s3dis/ranger/attention/CBAM/5batches_4ways_2shots_500epchoes/checkpoints/best_model.pth')
	elif(attention_type=='noAttention'):
		pretrain_state_dict = torch.load('/media/g111056119/SP PHD U3/datasets/log/pointnet/s3dis/ranger/8batches_4ways_2shots_500epchoes/checkpoints/best_model.pth')
	elif(attention_type=='NONE'): # don't load pretrain model
		pretrain_state_dict=None
	else:
		pretrain_state_dict=None
	return pretrain_state_dict
def main(args):
	meta_batchsz = args.meta_batchsz #8
	n_way = args.n_way
	k_shot = args.k_shot
	k_query = k_shot
	meta_lr = args.meta_lr
	num_updates = args.num_updates
	# [scSE,CBAM,ECANet,SENet,noAttention,NONE]
	attention_type = args.type
	dataset = args.dataset
	NUM_POINT = args.npoint
	print(f"train paremeters: {meta_batchsz} batches/ {n_way} ways / {k_shot} shots / {k_query} query / {num_updates} num_updates / {meta_lr} meta_lr  /{args.epoch} epchoes / {args.type} attention")
	
	'''CREATE DIR'''
	timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
	# experiment_dir = Path('./log/')
	experiment_dir = Path('/media/g111056119/SP PHD U3/datasets/log/')

	experiment_dir.mkdir(exist_ok=True)
	experiment_dir = experiment_dir.joinpath('pointnet')
	experiment_dir.mkdir(exist_ok=True)
	if args.log_dir is None:
		experiment_dir = experiment_dir.joinpath(timestr)
	else:
		experiment_dir = experiment_dir.joinpath(args.log_dir)
	experiment_dir.mkdir(exist_ok=True)
	checkpoints_dir = experiment_dir.joinpath('checkpoints/')
	checkpoints_dir.mkdir(exist_ok=True)

	if(dataset=='S3DIS' or dataset=='s3dis'):
		npy_root = '/media/g111056119/SP PHD U3/datasets/S3DIS/npy_13classes/'
		class2Label_dict = get_class2Label('datasets/S3DIS/meta/s3dis_classnames.txt')
		train_dataset = 'S3DIS'
		test_dataset = 'S3DIS'
		NUM_CLASSES_in = 13
		NUM_CLASSES_out = 13
		pretrain_state_dict = None
		db = S3DISDatasetFewShot(split='train', data_root=npy_root, num_point=NUM_POINT, block_size=1.0,num_episode=1000, n_way=n_way, k_shot=k_shot, n_queries=k_query, num_class=NUM_CLASSES_out)
		db_test = S3DISDatasetFewShot(split='test', data_root=npy_root, num_point=NUM_POINT, block_size=1.0, num_episode=500, n_way=n_way, k_shot=k_shot, n_queries=k_query, num_class=NUM_CLASSES_out)
		
	elif(dataset=='Scannet'):
		npy_root='/media/g111056119/SP PHD U3/datasets/ScanNet/scenes/data_21/'
		class2Label_dict = get_class2Label('datasets/ScanNet/meta/scannet_classnames_yun.txt')
		train_dataset = 'ScanNet'
		test_dataset = 'ScanNet'
		
		pretrain_state_dict = prtrain_model(attention_type)
		if(pretrain_state_dict==None):# no pretain
			NUM_CLASSES_in = 21
			NUM_CLASSES_out = 21
		else:# has pretrain
			NUM_CLASSES_in = 13
			NUM_CLASSES_out = 21
		db = ScannetDatasetFewShot(split='train', data_root=npy_root, num_point=NUM_POINT, block_size=1.0,num_episode=1000, n_way=n_way, k_shot=k_shot, n_queries=k_query, num_class=NUM_CLASSES_out)
		db_test = ScannetDatasetFewShot(split='test', data_root=npy_root, num_point=NUM_POINT, block_size=1.0, num_episode=500, n_way=n_way, k_shot=k_shot, n_queries=k_query, num_class=NUM_CLASSES_out)

	elif(dataset=='PersonS3DIS'):
		npy_root = '/media/g111056119/SP PHD U3/datasets/S3DIS/npy_14classes/'
		class2Label_dict = get_class2Label('datasets/S3DIS/meta/s3dis_classnames_person.txt')
		train_dataset = 'PersonS3DIS'
		test_dataset = 'PersonS3DIS'
		pretrain_state_dict = prtrain_model(attention_type)
		if(pretrain_state_dict==None):# no pretain
			NUM_CLASSES_in = 14
			NUM_CLASSES_out = 14
		else:# has pretrain
			NUM_CLASSES_in = 13
			NUM_CLASSES_out = 14
		db = S3DISDatasetFewShot(split='train', data_root=npy_root, num_point=NUM_POINT, block_size=1.0,num_episode=1000, n_way=n_way, k_shot=k_shot, n_queries=k_query, num_class=NUM_CLASSES_out)
		db_test = S3DISDatasetFewShot(split='test', data_root=npy_root, num_point=NUM_POINT, block_size=1.0, num_episode=500, n_way=n_way, k_shot=k_shot, n_queries=k_query, num_class=NUM_CLASSES_out)
	else:
		print("no this dataset")
		raise  NotImplementedError
	print(f"Meta train npy : {npy_root} train dataset : {train_dataset}")
	print(f"Meta test npy : {npy_root} test dataset : {test_dataset}")
	
	weights = torch.Tensor(db.labelweights).cuda()
	meta = MetaLearner(get_model, (NUM_CLASSES_in, attention_type), num_point=NUM_POINT, num_class_in=NUM_CLASSES_in, num_class_out=NUM_CLASSES_out, Label_dict=class2Label_dict, n_way=n_way, k_shot=k_shot, meta_batchsz=meta_batchsz, beta=meta_lr,
					num_updates=num_updates,weights=weights,pretrain_state_dict=pretrain_state_dict,epcho=args.epoch).cuda()

	tb = SummaryWriter('runs')
	# training sample : meta_batchsz(一個batch 有幾個episode) * 101 (epchoes)* [nway * (kshot+k_query) >> one episode size]
	# main loop 
	start_time = time.time()
	best_iou = 0

	for epchoes in range(args.epoch+1):
		print(f"-------------------episode_num = {epchoes} -------------------")
		# 1. train
		support_ptclouds_task=[]
		support_labels_task=[]
		query_ptclouds_task=[]
		query_labels_task=[]
		for i in range(meta_batchsz):
			get_one_episode = db[0]
			support_ptclouds, support_labels, query_ptclouds, query_labels = get_one_episode[0],get_one_episode[1],get_one_episode[2],get_one_episode[3]            
			support_ptclouds=torch.tensor(support_ptclouds, dtype=torch.float32)
			query_ptclouds=torch.tensor(query_ptclouds, dtype=torch.float32)
			support_labels=torch.tensor(support_labels, dtype=torch.int64)
			query_labels=torch.tensor(query_labels, dtype=torch.int64)

			# 點雲預處理
			support_ptclouds, support_labels = preprocess_point(support_ptclouds, support_labels)
			query_ptclouds, query_labels = preprocess_point(query_ptclouds, query_labels)
			support_ptclouds_task.append(support_ptclouds)
			support_labels_task.append(support_labels)
			query_ptclouds_task.append(query_ptclouds)
			query_labels_task.append(query_labels)

		# backprop has been embeded in forward func.
		# meta= MetaLearner()
		accs = meta(support_ptclouds_task, support_labels_task, query_ptclouds_task, query_labels_task, mode="train")
		train_acc = np.array(accs).mean()

		# 2. test
		if epchoes % 10 == 0:
			test_accs = []
			for i in range(10): # get average acc.
				support_ptclouds_task=[]
				support_labels_task=[]
				query_ptclouds_task=[]
				query_labels_task=[]
				for i in range(meta_batchsz):
					get_one_episode = db_test[0]
					support_ptclouds, support_labels, query_ptclouds, query_labels = get_one_episode[0],get_one_episode[1],get_one_episode[2],get_one_episode[3]            
					support_ptclouds=torch.tensor(support_ptclouds, dtype=torch.float32)
					query_ptclouds=torch.tensor(query_ptclouds, dtype=torch.float32)
					support_labels=torch.tensor(support_labels, dtype=torch.int64)
					query_labels=torch.tensor(query_labels, dtype=torch.int64)

					# 點雲預處理
					support_ptclouds, support_labels = preprocess_point(support_ptclouds, support_labels)
					query_ptclouds, query_labels = preprocess_point(query_ptclouds, query_labels)
					support_ptclouds_task.append(support_ptclouds)
					support_labels_task.append(support_labels)
					query_ptclouds_task.append(query_ptclouds)
					query_labels_task.append(query_labels)
 
				# get accuracy
				test_acc,best_iou= meta.pred(epchoes,support_ptclouds_task, support_labels_task, query_ptclouds_task, query_labels_task, mode="test", checkpoints_dir=checkpoints_dir, best_iou=best_iou)
				test_accs.append(test_acc)

			test_acc = np.array(test_accs).mean()
			print('epcho:', epchoes, '\t train acc:%.6f' % train_acc, '\t\ttest acc:%.6f' % test_acc,'\t\tbest_iou:%.6f' % best_iou)
			tb.add_scalar('test-acc', test_acc, epchoes)
			tb.add_scalar('train-acc', train_acc, epchoes)

	end_time = time.time()
	# Calculate the elapsed time
	elapsed_time = end_time - start_time

	# Convert elapsed time to a more readable format (optional)
	hours, remainder = divmod(elapsed_time, 3600)
	minutes, seconds = divmod(remainder, 60)

	# Print the training time
	print(f"Training time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")


if __name__ == '__main__':
	args = parse_args()
	main(args)
