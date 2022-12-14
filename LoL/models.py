import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.models import resnet

import optuna
from optuna.trial import TrialState

def define_model(trial, num_classes, num_feats):

	'''
	Function to create model to perform a optuna hyperparam sweep
	'''

	# We optimize the number of layers, hidden units and dropout ratio in each layer.
	n_layers = trial.suggest_int("n_layers", 1, 3)
	layers = []

	in_features = num_feats

	for i in range(n_layers):
		out_features = trial.suggest_int("n_units_l{}".format(i), 8, 64, step=8)
		layers.append(nn.Linear(in_features, out_features))
		layers.append(nn.ReLU())
		p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5, step=0.1)
		layers.append(nn.Dropout(p))
		in_features = out_features

	layers.append(nn.Linear(in_features, num_classes))
	layers.append(nn.LogSoftmax(dim=1))

	return nn.Sequential(*layers)

class BaseNet(nn.Module):

	def __init__(self, num_inputs, num_outputs):
		'''
		Constructor for an logistic regression model
		'''

		super(BaseNet, self).__init__()
		self.out_layer = nn.Linear(num_inputs, num_outputs)

	def forward(self, x):
		'''
		Calling a linear model
		'''

		return self.out_layer(x)
		# return torch.sigmoid(self.out_layer(x))

	def predict(self, x):
		'''
		Method to predict with a linear model
		'''

		preds = self.forward(x)
		return torch.max(preds, 1)[1].detach().numpy()

	def logit(self, x):
		return self.out_layer(x)

class TwoLayerNet(nn.Module):

	def __init__(self, num_inputs, num_outputs):
		'''
		Constructor for an two layer NN model
		'''

		super(TwoLayerNet, self).__init__()
		self.inner_layer = nn.Linear(num_inputs, 5)
		self.out_layer = nn.Linear(5, 2)
		self.dropout = nn.Dropout(p=0.3)
	def forward(self, x):
		'''
		Calling a linear model
		'''

		# return F.sigmoid(self.out_layer(self.dropout(F.relu(self.inner_layer(x)))))
		return self.out_layer(self.dropout(F.relu(self.inner_layer(x))))

	def logit(self, x):
		return self.out_layer(F.relu(self.inner_layer(x)))

	def predict(self, x):
		'''
		Method to predict with a linear model
		'''

		preds = self.forward(x)
		return torch.max(preds, 1)[1].detach().numpy()

class MLP(nn.Module):

	'''
	Class for a 3 Layer MLP
	'''
	
	def __init__(self, num_inputs, num_outputs):
		'''
		Constructor for an 3 Layer MLP
		'''

		super(MLP, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(num_inputs, 64),
			nn.ReLU(), 
			nn.Dropout(0.2),
			nn.Linear(64, 16),
			nn.ReLU(), 
			nn.Dropout(0.2),
			nn.Linear(16, num_outputs)
		)
		
	def forward(self, x):
		'''
		Calling a linear model
		'''

		return self.network(x)

	def predict(self, x):
		'''
		Method to predict with a linear model
		'''

		preds = self.forward(x)
		return torch.max(preds, 1)[1].detach().numpy()

class FineResnetModel(nn.Module):

	def __init__(self, num_outputs):
		'''
		Constructor for an fine-tuned resnet18
		'''

		super(FineResnetModel, self).__init__()
		self.dropout_rate = 0.5

		# setting up architecture for end model
		self.pretrained = resnet.resnet18(pretrained=True)
		for param in self.pretrained.parameters():
			param.requires_grad = False

		num_features = self.pretrained.fc.in_features
		self.pretrained.fc = nn.Linear(num_features, 49)
		self.out_layer = nn.Linear(49, num_outputs)
		self.dropout_layer = nn.Dropout(self.dropout_rate)

	def forward(self, images):
		'''
		Method for a foward pass
		'''

		pretrained_out = self.dropout_layer(F.relu(self.pretrained(images)))
		return self.out_layer(pretrained_out)

	def predict(self, x):
		'''
		Method to predict with a resnet
		'''
		preds = self.forward(x)
		return torch.max(preds, 1)[1].detach().numpy()

	def logit(self, images):
		pretrained_out = self.dropout_layer(F.relu(self.pretrained(images)))
		return self.out_layer(pretrained_out)		


class LSTMModel(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, vocab_size, num_outputs):
		super(LSTMModel, self).__init__()
		self.hidden_dim = hidden_dim
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

		# The LSTM takes word embeddings as inputs, and outputs hidden states
		# with dimensionality hidden_dim.
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

		# The linear layer that maps from hidden state space to prediction
		self.linear = nn.Linear(hidden_dim, num_outputs)

	def forward(self, sentence):
		embeds = self.word_embeddings(sentence)
		lstm_out, (ht, ct) = self.lstm(embeds)
		return self.linear(ht[-1])
	
	def predict(self, sentence):
		preds = self.forward(sentence)
		return torch.max(preds, 1)[1].detach().numpy()