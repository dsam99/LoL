import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.autograd import Variable
from torch import autograd
import argparse
import sys
import pickle
from sklearn.model_selection import train_test_split

def LoL_simple(model_out, votes, device, cov_weights, weights=None):
	
	'''
	Automatically generating losses from weak labelers without gradient information
	
	Args:
	model_out - output of the model
	votes - weak supervision source votes
	device - pytorch device
	cov_weights - coverage weights for each weak supervision vote
	weights - additional weight to combine losses (i.e. from snorkel or small set of labeled data)
	'''

	criterion = nn.CrossEntropyLoss(reduction="none")
	loss = 0

	valid_inds = [torch.where(votes[:,i] != -1, 1, 0).nonzero().flatten() for i in range(votes.shape[1])]
	data_weighting = torch.zeros(votes.shape[0])

	for l in valid_inds:
		data_weighting[l] += 1
	data_weighting = 1 / data_weighting

	# don't correct for data weighting
	data_weighting = data_weighting.to(device)

	for i in range(np.shape(votes)[1]):
		if valid_inds[i].shape[0] > 0:
			l = criterion(model_out[valid_inds[i]], votes[valid_inds[i], i]) 

			# weight losses equally
			if weights is None:
				loss += (torch.sum(l * data_weighting[valid_inds[i]])) * cov_weights[i] / votes.shape[1]
			
			# weight based on some prior notion
			else:
				loss += (torch.sum(l * data_weighting[valid_inds[i]])) * cov_weights[i] * weights[i]
	return loss

def LoL(model, data, votes, rand_data, indices, device, cov_weights, optim, weights=None, num_classes=2, grad_val=1, alpha=0.01, method="square"):
	'''
	Automatically generating loss from weak labelers outputs + gradients

	Args:
		model - model to train
		data - data to train model on
		votes - weak supervision source outputs (here denoted as hard votes)
		rand_data - random sample of data to match gradients of model on
		indices - list of dictionaries of the form {class -> indices} where indices are features that the weak supervision sources care about
		device - torch device
		num_classes - num_classes in the data
		grad_val - value to match gradients on random data to
		alpha - amount to weight the gradient component of the loss
		cov_weights - coverage weights to combine losses for each weak supervision vote
		optim - optimizer (required only to zero grad again when computing gradient)
	Returns
		value representing gradloss value
	'''

	criterion = nn.CrossEntropyLoss(reduction="none")
	loss = 0 

	valid_inds = [torch.where(votes[:,i] != -1, 1, 0).nonzero().flatten() for i in range(votes.shape[1])]

	# computing data weighting to treat data w same importance
	data_weighting = torch.zeros(votes.shape[0])
	for l in valid_inds:
		data_weighting[l] += 1
	data_weighting = 1 / data_weighting
	data_weighting = data_weighting.to(device)

	# computing gradient information
	rand_data.requires_grad = True
	# jac = jacobian(model, rand_data, create_graph=True)
	# grad = jac.mean(dim=(0, 2))
	
	rand_out = model(rand_data)
	rand_out.backward(gradient=rand_out)
	grad = rand_data.grad
	optim.zero_grad()
	
	preds = model(data)

	for i in range(np.shape(votes)[1]):

		# if not abstaining on all points
		if valid_inds[i].shape[0] > 0:
			l = criterion(preds[valid_inds[i]], votes[valid_inds[i],i])
			
			# if grad information available
			if indices[i] != []:
				loss_grad = 0        
				for key_val in list(indices[i].keys()):
					if method == "square":
						loss_grad += torch.sum(torch.square(grad[key_val, indices[i][key_val]] - grad_val))  
					elif method == "exp":
						loss_grad += torch.sum(torch.exp(torch.clamp(grad[key_val, indices[i][key_val]] - grad_val, max=0)))     
					elif method == "linear":
						loss_grad += torch.sum(torch.abs(torch.clamp(grad[key_val, indices[i][key_val]] - grad_val, max=0)))     
				to_add = (torch.sum(l * data_weighting[valid_inds[i]]) + alpha * loss_grad) * cov_weights[i] 
			else:
				to_add = torch.sum(l * data_weighting[valid_inds[i]]) * cov_weights[i]
		
			if weights is None:
				to_add /= votes.shape[1]
			else:
				to_add *= weights[i]
			loss += to_add

	return loss


def LoL_awa2(model, data, votes, wl_grad, device, cov_weights, weights=None, num_classes=2, grad_val=1, alpha=0.01, method="square"):
	'''
	Automatically generating loss from weak labelers outputs + gradients (specific implementation for AwA2 experiments)

	Args:
		model - model to train
		data - data to train model on
		votes - weak supervision source outputs (here denoted as hard votes)
		rand_data - random sample of data to match gradients of model on
		indices - list of dictionaries of the form {class -> indices} where indices are features that the weak supervision sources care about
		device - torch device
		num_classes - num_classes in the data
		grad_val - value to match gradients on random data to
		alpha - amount to weight the gradient component of the loss
		cov_weights - coverage weights to combine losses for each weak supervision vote

	Returns
		value representing gradloss value
	'''
	criterion = nn.CrossEntropyLoss(reduction="none")
	loss = 0 

	valid_inds = [torch.where(votes[:,i] != -1, 1, 0).nonzero().flatten() for i in range(votes.shape[1])]

	# computing data weighting to treat data w same importance
	data_weighting = torch.zeros(votes.shape[0])
	for l in valid_inds:
		data_weighting[l] += 1
	data_weighting = 1 / data_weighting
	data_weighting = data_weighting.to(device)

	rand_data.requires_grad = True
	jac = jacobian(model, rand_data, create_graph=True)
	grad = jac.mean(dim=(0, 2))
	preds = model(data)

	for i in range(np.shape(votes)[1]):

		# if not abstaining on all points
		if valid_inds[i].shape[0] > 0:
			l = criterion(preds[valid_inds[i]], votes[valid_inds[i],i])
				
			loss_grad = torch.sum(torch.square(grad - wl_grad))  
			to_add = (torch.sum(l * data_weighting[valid_inds[i]]) + alpha * loss_grad) * cov_weights[i] 
		
			if weights is None:
				to_add /= votes.shape[1]
			else:
				to_add *= weights[i]
			loss += to_add

	return loss