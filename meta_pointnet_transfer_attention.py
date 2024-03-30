import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

from ranger21 import Ranger21  # this is from ranger.py
# from ranger import RangerVA  # this is from ranger913A.py
# from ranger import RangerQH  # this is from rangerqh.py


class Learner(nn.Module):
	"""
	This is a learner class, which will accept a specific network module, such as OmniNet that define the network forward
	process. Learner class will create two same network, one as theta network and the other acts as theta_pi network.
	for each episode, the theta_pi network will copy its initial parameters from theta network and update several steps
	by meta-train set and then calculate its loss on meta-test set. All loss on meta-test set will be sumed together and
	then backprop on theta network, which should be done on metalaerner class.
	For learner class, it will be responsible for update for several steps on meta-train set and return with the loss on
	meta-test set.
	"""
	def __init__(self, net_cls, *args, num_point, num_class_in, num_class_out, pretrain_state_dict, meta_batchsz, epcho):
		"""
		It will receive a class: net_cls and its parameters: args for net_cls.
		:param net_cls: class, not instance
		:param args: the parameters for net_cls
		"""
		super(Learner, self).__init__()
		print("Learner __init__")
		# pls make sure net_cls is a class but NOT an instance of class.
		# print("net_cls.__class__=",net_cls.__class__)
		assert net_cls.__class__ == type

		self.NUM_CLASSES = num_class_out
		self.NUM_POINT = num_point
		classifier = net_cls(*args).cuda()
		classifier.apply(self.inplace_relu)
		# classifier = classifier.apply(self.weights_init)

		# Load pre-trained state_dict if provided
		if pretrain_state_dict is not None:
			# 透過修改model.classifier.conv2，使輸出的channel符合自己Task的需求，例如out_ch。
			classifier.conv2 = nn.Conv1d(128, num_class_in, 1)
			classifier.load_state_dict(pretrain_state_dict['model_state_dict'])
			classifier.conv2 = nn.Conv1d(128, num_class_out, 1)
			# print("classifier = ", classifier)

		# we will create two class instance meanwhile and use one as theta network and the other as theta_pi network.
		self.net = classifier
		from models import pointnet2_sem_seg_msg_attention as MODEL
		self.criterion = MODEL.get_loss().cuda()
		# you must call create_pi_net to create pi network additionally
		self.net_pi = classifier
		# self.optimizer = optim.SGD(self.net_pi.parameters(), 0.001,  momentum=0.9)
		self.optimizer = Ranger21(self.net_pi.parameters(), lr=0.001,using_gc=True,gc_conv_only=False,num_batches_per_epoch=meta_batchsz,num_epochs=epcho)

	def inplace_relu(self,m):
		classname = m.__class__.__name__
		if classname.find('ReLU') != -1:
			m.inplace=True
	def weights_init(self,m):
		classname = m.__class__.__name__
		if classname.find('Conv2d') != -1:
			torch.nn.init.xavier_normal_(m.weight.data)
			torch.nn.init.constant_(m.bias.data, 0.0)
		elif classname.find('Linear') != -1:
			torch.nn.init.xavier_normal_(m.weight.data)
			torch.nn.init.constant_(m.bias.data, 0.0)
	def parameters(self):
		"""
		Override this function to return only net parameters for MetaLearner's optimize
		it will ignore theta_pi network parameters.
		:return:
		"""
		return self.net.parameters()
	def net_pi_state_dict(self):
		"""
		return net_pi state_dict to MetaLearner pred function
		:return:
		"""
		return self.net_pi.state_dict(),self.optimizer.state_dict()

	def update_pi(self):
		"""
		copy parameters from self.net -> self.net_pi
		:return:
		"""
		print("Learner update_pi")
		for m_from, m_to in zip(self.net.modules(), self.net_pi.modules()):
			if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d) or isinstance(m_to, nn.Conv1d) or isinstance(m_to, nn.BatchNorm1d):
				m_to.weight.data = m_from.weight.data.clone()
				if m_to.bias is not None:
					m_to.bias.data = m_from.bias.data.clone()

	def bn_momentum_adjust(self,m, momentum):
		if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
			m.momentum = momentum

	def forward(self, support_x, support_y, query_x, query_y, num_updates, weights):
		"""
		learn on current episode meta-train: support_x & support_y and then calculate loss on meta-test set: query_x&y
		:param support_x: [setsz, c_, h, w]
		:param support_y: [setsz]
		:param query_x:   [querysz, c_, h, w]
		:param query_y:   [querysz]
		:param num_updates: 5
		:return:
		"""
		print("Learner forward")

		# now try to fine-tune from current $theta$ parameters -> $theta_pi$
		# after num_updates of fine-tune, we will get a good theta_pi parameters so that it will retain satisfying
		# performance on specific task, that's, current episode.
		# firstly, copy theta_pi from theta network
		self.update_pi()
		total_correct = 0
		total_seen = 0
		loss_sum = 0
		setsz = support_x.shape[0] # = nway * kshot
		querysz = query_x.shape[0] # = nway * k_query
		"""
		1. Train update for several steps
		meta-train: support_x & support_y
		"""
		for i in range(num_updates):
			print(f"-------------------num_updates = {i} -------------------")
			# forward and backward to update net_pi grad.
			seg_pred, trans_feat = self.net_pi(support_x) 
			seg_pred = seg_pred.contiguous().view(-1, self.NUM_CLASSES)

			batch_label = support_y.view(-1, 1)[:, 0].cpu().data.numpy()
			support_y = support_y.view(-1, 1)[:, 0]
            
			loss = self.criterion(seg_pred, support_y, trans_feat, weights)
			print("Train loss_s=",loss)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
			correct = np.sum(pred_choice == batch_label)
			total_correct += correct
			total_seen += (setsz * self.NUM_POINT) 
			loss_sum += loss
		print("loss_sum_s",loss_sum)
		print('Training train episode mean loss: %f' % (loss_sum / num_updates))
		print('Training train episode accuracy: %f' % (total_correct / float(total_seen)))

		"""
		2.  calculate loss on meta-train's test set: query_x&y
		meta-train: query_x & query_y
		"""
		# Compute the meta gradient and return it, the gradient is from one episode
		# in metalearner, it will merge all loss from different episode and sum over it.
		seg_pred_q, trans_feat_q = self.net_pi(query_x)
		seg_pred_q = seg_pred_q.contiguous().view(-1, self.NUM_CLASSES)
		batch_label_q = query_y.view(-1, 1)[:, 0].cpu().data.numpy()
		query_y = query_y.view(-1, 1)[:, 0]
		
		loss = self.criterion(seg_pred_q, query_y, trans_feat_q, weights)
		print("Train loss_q=",loss)

		pred_choice_q = seg_pred_q.cpu().data.max(1)[1].numpy()
		correct_q = np.sum(pred_choice_q == batch_label_q)
		total_seen_q = (querysz * self.NUM_POINT) 
		accuracy = correct_q / float(total_seen_q)

		print('Training test episode loss: %f' % (loss))
		# print('Training mean loss: %f' % (loss_sum / num_batches))# num_batches=num_episode / batch_size = len(trainDataLoader)
		print('Training test episode accuracy: %f' % (accuracy))


		# gradient for validation on theta_pi
		# after call autorad.grad, you can not call backward again except for setting create_graph = True
		# as we will use the loss as dummpy loss to conduct a dummy backprop to write our gradients to theta network,
		# here we set create_graph to true to support second time backward.
		grads_pi = autograd.grad(loss, self.net_pi.parameters(), create_graph=True)

		return loss, grads_pi, accuracy

	def net_forward(self, support_x, support_y, weights):
		"""
		This function is purely for updating net network. In metalearner, we need the get the loss op from net network
		to write our merged gradients into net network, hence will call this function to get a dummy loss op.
		:param support_x: [setsz, c, h, w]
		:param support_y: [sessz, c, h, w]
		:return: dummy loss and dummy pred
		"""
		print("Learner net_forward")
		seg_pred, trans_feat = self.net(support_x)
		seg_pred = seg_pred.contiguous().view(-1,self.NUM_CLASSES)
		support_y = support_y.view(-1, 1)[:, 0]
		loss = self.criterion(seg_pred, support_y, trans_feat, weights)
		print("net_forward: loss_s=",loss)
		pred_choice = seg_pred.cpu().data.max(1)[1].numpy()

		return loss, pred_choice
	
	def pred(self, support_x, support_y, query_x, query_y, num_updates, weights):
		"""
		learn on current episode meta-test: support_x & support_y and then calculate loss on meta-test set: query_x & query_y
		:param support_x: [setsz, c_, h, w]
		:param support_y: [setsz]
		:param query_x:   [querysz, c_, h, w]
		:param query_y:   [querysz]
		:param num_updates: 5
		:return:
		"""
		print("Learner pred")

		# now try to fine-tune from current $theta$ parameters -> $theta_pi$
		# after num_updates of fine-tune, we will get a good theta_pi parameters so that it will retain satisfying
		# performance on specific task, that's, current episode.
		# firstly, copy theta_pi from theta network
		self.update_pi()
		total_correct = 0
		total_seen = 0
		loss_sum = 0
		setsz = support_x.shape[0] # = nway * kshot
		querysz = query_x.shape[0] # = nway * k_query
		"""
		1. Finetune Train update for several steps
		meta-test: support_x & support_y
		"""
		for i in range(num_updates):
			print(f"-------------------num_updates = {i} -------------------")
			# forward and backward to update net_pi grad.
			seg_pred, trans_feat = self.net_pi(support_x) 
			seg_pred = seg_pred.contiguous().view(-1, self.NUM_CLASSES)

			batch_label = support_y.view(-1, 1)[:, 0].cpu().data.numpy()
			support_y = support_y.view(-1, 1)[:, 0]
            
			loss = self.criterion(seg_pred, support_y, trans_feat, weights)
			print("loss_s=",loss)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
			correct = np.sum(pred_choice == batch_label)
			total_correct += correct
			total_seen += (setsz * self.NUM_POINT) 
			loss_sum += loss
		print("loss_sum_s",loss_sum)
		print('Testing train episode mean loss: %f' % (loss_sum / num_updates))
		print('Testing train episode accuracy: %f' % (total_correct / float(total_seen)))

		"""
		2.  calculate loss on meta-test's test set: query_x & query_y
		meta-test: query_x & query_y
		"""
		# Compute the meta gradient and return it, the gradient is from one episode
		# in metalearner, it will merge all loss from different episode and sum over it.
		self.net_pi = self.net_pi.eval()

		seg_pred_q, trans_feat_q = self.net_pi(query_x)
		pred_val_q = seg_pred_q.contiguous().cpu().data.numpy()
		seg_pred_q = seg_pred_q.contiguous().view(-1, self.NUM_CLASSES)

		batch_label_q = query_y.cpu().data.numpy()
		query_y = query_y.view(-1, 1)[:, 0]
		loss = self.criterion(seg_pred_q, query_y, trans_feat_q, weights)
		print("Test train loss_q=",loss)

		pred_val_q = np.argmax(pred_val_q, 2)
		correct_q = np.sum((pred_val_q == batch_label_q))

		total_seen_q = (querysz * self.NUM_POINT) 
		accuracy = correct_q / float(total_seen_q)

		tmp, _ = np.histogram(batch_label_q, range(self.NUM_CLASSES + 1))

		# gradient for validation on theta_pi
		# after call autorad.grad, you can not call backward again except for setting create_graph = True
		# as we will use the loss as dummpy loss to conduct a dummy backprop to write our gradients to theta network,
		# here we set create_graph to true to support second time backward.
		grads_pi = autograd.grad(loss, self.net_pi.parameters(), create_graph=True)
		model_state_dict = self.net_pi.state_dict(),
		optimizer_state_dict = self.optimizer.state_dict()

		return loss, grads_pi, accuracy, tmp, batch_label_q, pred_val_q, total_seen_q, model_state_dict, optimizer_state_dict

class MetaLearner(nn.Module):
	"""
	As we have mentioned in Learner class, the metalearner class will receive a series of loss on different tasks/episodes
	on theta_pi network, and it will merage all loss and then sum over it. The summed loss will be backproped on theta
	network to update theta parameters, which is the initialization point we want to find.
	"""

	def __init__(self, net_cls, net_cls_args, num_point, num_class_in, num_class_out ,Label_dict, n_way, k_shot, meta_batchsz, beta, num_updates, weights, pretrain_state_dict, epcho):
		"""

		:param net_cls: class, not instance. the class of specific Network for learner
		:param net_cls_args: tuple, args for net_cls, like (n_way, imgsz) single value use(num_classes,)
		:param n_way:
		:param k_shot:
		:param meta_batchsz: number of tasks/episode
		:param beta: learning rate for meta-learner
		:param num_updates: number of updates for learner
		"""
		super(MetaLearner, self).__init__()
		print("MetaLearner __init__")
		
		self.n_way = n_way
		self.k_shot = k_shot
		self.meta_batchsz = meta_batchsz
		self.beta = beta
		# self.alpha = alpha # set alpha in Learner.optimizer directly.
		self.num_updates = num_updates
		self.weights = weights
		self.NUM_POINT = num_point
		self.NUM_CLASSES = num_class_out
		self.Label_dict = Label_dict
		self.epcho = epcho
		# it will contains a learner class to learn on episodes and gather the loss together.
		self.learner = Learner(net_cls, *net_cls_args, num_point=num_point, num_class_in=num_class_in, num_class_out=num_class_out, pretrain_state_dict=pretrain_state_dict, meta_batchsz=meta_batchsz, epcho=epcho)

		# the optimizer is to update theta parameters, not theta_pi parameters.
		# self.optimizer = optim.Adam(self.learner.parameters(), lr=beta)
		self.optimizer = Ranger21(self.learner.parameters(), lr=beta,using_gc=True,gc_conv_only=False,num_batches_per_epoch=meta_batchsz,num_epochs=epcho)

	def write_grads(self, dummy_loss, sum_grads_pi):
		"""
		write loss into learner.net, gradients come from sum_grads_pi.
		Since the gradients info is not calculated by general backward, we need this function to write the right gradients
		into theta network and update theta parameters as wished.
		:param dummy_loss: dummy loss, nothing but to write our gradients by hook
		:param sum_grads_pi: the summed gradients
		:return:
		"""
		print("MetaLearner write_grads")
		# Register a hook on each parameter in the net that replaces the current dummy grad
		# with our grads accumulated across the meta-batch
		hooks = []

		for i, v in enumerate(self.learner.parameters()):
			def closure():
				ii = i
				return lambda grad: sum_grads_pi[ii]

			# if you write: hooks.append( v.register_hook(lambda grad : sum_grads_pi[i]) )
			# it will pop an ERROR, i don't know why?
			hooks.append(v.register_hook(closure()))

		# use our sumed gradients_pi to update the theta/net network,
		# since our optimizer receive the self.net.parameters() only.
		self.optimizer.zero_grad()
		dummy_loss.backward()
		self.optimizer.step()

		# if you do NOT remove the hook, the GPU memory will expode!!!
		for h in hooks:
			h.remove()

	def forward(self, support_x, support_y, query_x, query_y, mode:str):
		"""
		Here we receive a series of episode, each episode will be learned by learner and get a loss on parameters theta.
		we gather the loss and sum all the loss and then update theta network.
		setsz = n_way * k_shot
		querysz = n_way * k_shot
		:param support_x: [meta_batchsz, setsz, c_, h, w]
		:param support_y: [meta_batchsz, setsz]
		:param query_x:   [meta_batchsz, querysz, c_, h, w]
		:param query_y:   [meta_batchsz, querysz]
		:return:
		"""
		print("MetaLearner forward mode = ",mode)
		sum_grads_pi = None
		meta_batchsz = len(support_y)
		# we do different learning task sequentially, not parallel.
		accs = []
		# for each task/episode.
		for i in range(meta_batchsz):
			_, grad_pi, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates, self.weights)
			accs.append(episode_acc)
			if sum_grads_pi is None:
				sum_grads_pi = grad_pi
			else:  # accumulate all gradients from different episode learner
				sum_grads_pi = [torch.add(i, j) for i, j in zip(sum_grads_pi, grad_pi)]

		# As we already have the grads to update
		# We use a dummy forward / backward pass to get the correct grads into self.net
		# the right grads will be updated by hook, ignoring backward.
		# use hook mechnism to write sumed gradient into network.
		# we need to update the theta/net network, we need a op from net network, so we call self.learner.net_forward
		# to get the op from net network, since the loss from self.learner.forward will return loss from net_pi network.
		dummy_loss, _ = self.learner.net_forward(support_x[0], support_y[0], self.weights)
		self.write_grads(dummy_loss, sum_grads_pi)

		return accs

	def pred(self, epchoes, support_x, support_y, query_x, query_y, mode:str, checkpoints_dir:str, best_iou):
		"""
		predict for query_x
		setsz = n_way * k_shot
		querysz = n_way * k_query
		:param support_x: [meta_batchsz, setsz, c_, h, w]
		:param support_y: [meta_batchsz, setsz]
		:param query_x:   [meta_batchsz, querysz, c_, h, w]
		:param query_y:   [meta_batchsz, querysz]
		:param mode: test
		:return:
		"""
		# meta_batchsz = support_y.size(0)
		print("MetaLearner pred mode = ",mode)
		meta_batchsz = len(support_y)
		total_correct = 0
		querysz = query_x[0].shape[0] # = nway * k_query
		# total_seen_q = (querysz * self.NUM_POINT) * len(query_y)
		total_seen=0
		loss_sum = 0
		labelweights = np.zeros(self.NUM_CLASSES)
		total_seen_class = [0 for _ in range(self.NUM_CLASSES)]
		total_correct_class = [0 for _ in range(self.NUM_CLASSES)]
		total_iou_deno_class = [0 for _ in range(self.NUM_CLASSES)]
		accs = []
		
		# for each task/episode.
		# the learner will copy parameters from current theta network and then fine-tune on support set.
		
		for i in range(meta_batchsz):
			loss, _, episode_acc, tmp, batch_label_q, pred_val_q, total_seen_q, _, _= self.learner.pred(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates, self.weights)
			accs.append(episode_acc)
			loss_sum+=loss
			total_correct+=episode_acc
			labelweights += tmp
			total_seen += total_seen_q
			for l in range(self.NUM_CLASSES):
				total_seen_class[l] += np.sum((batch_label_q == l))
				total_correct_class[l] += np.sum((pred_val_q == l) & (batch_label_q == l))
				total_iou_deno_class[l] += np.sum(((pred_val_q == l) | (batch_label_q == l)))

		labelweights = labelweights.astype(float) / np.sum(labelweights.astype(float))
		mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
		print(f'---- EPOCH {epchoes} EVALUATION ----')
		print('Eval mean loss: %f' % (loss_sum / float(meta_batchsz)))
		print('Eval point avg class IoU: %f' % (mIoU))
		print('Eval point accuracy: %f' % (total_correct / float(total_seen_q)))
		print('Eval point avg class acc: %f' % (
			np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))))
		iou_per_class_str = '------- IoU --------\n'
		for l in range(self.NUM_CLASSES):
			iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
				self.Label_dict[l] + ' ' * (14 - len(self.Label_dict[l])), labelweights[l - 1],
				total_correct_class[l] / (float(total_iou_deno_class[l] + 1e-6))) # 1e-6: avoid divide zero cause NAN
		model_state_dict_,optimizer_state_dict_=self.learner.net_pi_state_dict()
		print(iou_per_class_str)
		if mIoU >= best_iou:
			best_iou = mIoU
			print('Save model...')
			savepath = str(checkpoints_dir) + '/best_model.pth'
			print('Saving at %s' % savepath)
			state = {
				'epoch': meta_batchsz,
				'class_avg_iou': mIoU,
				'model_state_dict': model_state_dict_,
				'optimizer_state_dict': optimizer_state_dict_,
			}
			torch.save(state, savepath)
		print('Best mIoU: %f' % best_iou)
		print('Save model last step modle state...')
		savepath = str(checkpoints_dir) + '/lastmodel.pth'
		print('Saving at %s' % savepath)
		state = {
			'epoch': meta_batchsz,
			'class_avg_iou': mIoU,
			'model_state_dict': model_state_dict_,
			'optimizer_state_dict': optimizer_state_dict_,
		}
		torch.save(state, savepath)

		return np.array(accs).mean(), best_iou