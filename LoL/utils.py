import torch
import numpy as np
import torch.nn as nn

def bce_prob_loss(preds, soft_label):
	'''
	Function to compute binary cross entropy loss with soft labels
	'''
	return -1 * torch.mean(soft_label * torch.log(preds) + (1 - soft_label) * torch.log(1 - preds))

def entropy(prob_dist):
	'''
	Function to compute the entropy of a given probability distribution
	'''
	return -np.mean(prob_dist * np.log(prob_dist) + (1 - prob_dist) * np.log(1 - prob_dist))

def torch_bce(model_out, vote):
	'''
	Version of pytorch binary cross entropy with log values clipped to help with numerical overflow
	'''
	# clipping log values
	return -1 * (torch.reshape(vote, (-1, 1)) * torch.clamp(torch.log(model_out), min=-100) + torch.reshape((1 - vote), (-1, 1)) * torch.clamp(torch.log(1 - model_out), min=-100))    

def random_sample(num_data, data_size, p=0.01):
	'''
	Sampling data w a certain probability (of size x) iid from a Bernoulli distribution with parameter p
	'''

	rand_samp = torch.zeros((num_data, data_size))
	for i in range(num_data):
		for j in range(data_size):
			r = np.random.uniform(0, 1)
			if r <= p:
				rand_samp[i,j] = 1
	return rand_samp