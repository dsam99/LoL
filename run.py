import argparse
import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets

from data.process_data import load_split_data, create_dataloader
from LoL.losses_over_labels import LoL, LoL_simple
from LoL.models import MLP, define_model
from LoL.utils import random_sample

from snorkel.labeling.model import MajorityLabelVoter, LabelModel
from snorkel.classification import cross_entropy_with_probs
from torchvision import transforms

import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
import gc

optuna.logging.set_verbosity(optuna.logging.WARNING)
print("GPU", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default="youtube", type=str, help="Dataset to run (spam, agnews, yelp, awa2)")
	parser.add_argument('--LoL_simple', action='store_true', help="Flag to run our method of LoL-simple")
	parser.add_argument('--LoL', action='store_true', help="Flag to run our method of LoL")
	parser.add_argument("--cov", action='store_true', help="If using weighting by coverage (no by default)")
	parser.add_argument("--sw", action='store_true', help="If using snorkel weighting")
	parser.add_argument("--grad_method", default="square", type=str, help="Which gradient penalty to use")
	
	parser.add_argument("--mv", action='store_true', help="Run Majority Vote baseline")
	parser.add_argument('--snorkel', action='store_true', help="Flag to run Snorkel Baseline")
	parser.add_argument("--soft", action='store_true', help="Run snorkel with soft labeling")
	
	args = parser.parse_args()

	# seeds = [0, 1, 2, 3, 4]
	seeds = [0]
	num_class_dict = {"youtube":2, "yelp":2, "awa2":2, "agnews":4, "chemprot":10, "imdb":2}
	num_classes = num_class_dict[args.dataset]
	
	print("Dataset: ", args.dataset)
	if args.snorkel:
		print("Snorkel (MeTaL)")
	if args.LoL_simple:
		print("LoL-simple")
		print("Cov", args.cov)
	if args.LoL:
		grad_method = args.grad_method
		print("LoL", grad_method)
		print("Cov", args.cov)
	
	if args.LoL or args.LoL_simple:
		if args.sw:
			print("Snorkel weighting")

	res = []

	num_epochs = 30
	print("Num Epochs", num_epochs)
	batch_size = 128

	# run over 5 seeds
	for seed in seeds:
		print("Seed", seed)
		random.seed(seed)
		os.environ['PYTHONHASHSEED'] = str(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = True
		
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)

		# extra runs to fit grad_val parameter
		if args.LoL:
			num_trials = 30
		else:
			num_trials = 15

		test_accs = np.zeros(num_trials)

		# loading split data
		(train_data, train_votes, train_labels), (val_data, val_votes, val_labels), \
			(test_data, test_votes, test_labels), lf_inds = load_split_data(args.dataset, seed, 'wrench')
		num_feats = train_data.shape[1]
		
		# filtering out data with all abstain votes using Snorkel (MeTaL)
		label_model = LabelModel(cardinality=num_classes, verbose=False)
		label_model.fit(L_train=train_votes, n_epochs=200, log_freq=200, seed=seed)
		pseudolabs = label_model.predict(train_votes)
		valid_inds = np.where(pseudolabs != -1)[0]
		
		train_data, train_votes, train_labels = train_data[valid_inds], train_votes[valid_inds], train_labels[valid_inds]
		val_loader = create_dataloader(val_data, val_labels, shuffle=False)
		test_loader = create_dataloader(test_data, test_labels, shuffle=False)

		# train model on weak supervision
		if args.snorkel or args.mv:
			lm_test = label_model.predict(test_votes)

			if args.mv:
				mv_model = MajorityLabelVoter(cardinality=num_classes)
				majority_acc = mv_model.score(test_votes, test_labels, tie_break_policy="random")["accuracy"]	
				pseudolabs = mv_model.predict(train_votes, tie_break_policy="true-random")
			elif args.snorkel:
				pseudolabs = label_model.predict_proba(train_votes)
			
			# defining loss function for snorkel baseline
			ws_obj = nn.CrossEntropyLoss()
			wrench_loader = data.DataLoader(data.TensorDataset(torch.tensor(train_data, dtype=torch.float), 
				torch.tensor(pseudolabs, dtype=torch.float)), 
				batch_size=batch_size, shuffle=True)

			def snorkel_objective(trial):
				'''
				Function for Optuna to define Snorkel (MeTaL)'s training objective
				'''

				model = MLP(num_feats, num_classes).to(device)
				lr = trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2, 1e-1])
				l2 = trial.suggest_categorical("l2", [0, 1e-2, 1e-1])
				optimizer = optim.Adam(model.parameters(), weight_decay=l2, lr=lr)
				if not args.wrench:
					batch_size = trial.suggest_int("batch_size", 100, 300, step=100)
								
					if args.soft:
						# creating trainloader with soft labels
						train_loader = data.DataLoader(data.TensorDataset(torch.tensor(train_data, dtype=torch.float), 
													torch.tensor(pseudolabs, dtype=torch.float)), 
													batch_size=batch_size, shuffle=True)
					else:
						train_loader = create_dataloader(train_data, pseudolabs, batch_size=batch_size, shuffle=True)
				else:
					train_loader = wrench_loader
				trial_val_accs = np.zeros(num_epochs)
				trial_test_accs = np.zeros(num_epochs)

				for i in range(num_epochs): 
					model.train()
					for x, vote in train_loader:
						x, vote = x.to(device), vote.to(device)
						optimizer.zero_grad()
						outputs = model(x)
						if args.soft:
							# computing soft cross entropy loss
							loss = cross_entropy_with_probs(outputs, vote)
						else:
							loss = ws_obj(outputs, vote)
							
						loss.backward()
						optimizer.step()

					# Validation of the model
					model.eval()
					val_preds = []
					test_preds = []
					with torch.no_grad():
						for x, _ in val_loader:
							x = x.to(device)
							ws_preds = torch.max(model(x), dim=1)[1].cpu().detach().numpy()
							val_preds.append(ws_preds)
					
						for x, _ in test_loader:
							x = x.to(device)
							auto_preds = torch.max(model(x), dim=1)[1].cpu().detach().numpy()
							test_preds.append(auto_preds)

					val_preds = np.concatenate(val_preds)
					val_accuracy = np.mean(val_preds == val_labels)
					test_preds = np.concatenate(test_preds)
					test_accuracy = np.mean(test_preds == test_labels)
					
					trial_val_accs[i] = val_accuracy
					trial_test_accs[i] = test_accuracy

					# Handle pruning based on the intermediate value every 3 epochs
					if i % 3 == 0:
						trial.report(val_accuracy, i)			
						if trial.should_prune():
							raise optuna.exceptions.TrialPruned()
				
				# getting best val & test accuracy over epochs
				best_val_accuracy, best_ind = np.max(trial_val_accs), np.argmax(trial_val_accs)
				best_test_accuracy = trial_test_accs[best_ind]
				
				# storing best test accuracy for later
				test_accs[trial.number] = best_test_accuracy				
				return best_val_accuracy
			
			sampler = TPESampler(seed = seed) 
			study = optuna.create_study(sampler=sampler, direction="maximize")
			study.optimize(snorkel_objective, n_trials=num_trials, timeout=6000, gc_after_trial=True)

			pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
			complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
			trial = study.best_trial

			if args.mv:
				print("MV EM Test Accuracy", test_accs[trial.number])
			else:
				print("Snorkel MeTaL EM Test Accuracy", test_accs[trial.number])
			
			res.append(test_accs[trial.number])

		# run our LoL approach
		else:

			# num of random examples to compute gradient on
			num_rand = 10

			# computing coverages and corresponding weights
			if args.cov:
				prop_weights = torch.zeros(train_votes.shape[1])
				for i in range(train_votes.shape[1]):
					valid_num = np.sum(train_votes[:,i] != -1)
					prop_weights[i] = (1 / valid_num)
			else:
				prop_weights = torch.ones(train_votes.shape[1]) / train_votes.shape[0]
			
			# weighting schemes
			weights = None

			if args.sw:
				sw = label_model.get_weights()
				weights = sw / np.sum(sw)			

			wrench_loader = create_dataloader(train_data, train_votes, batch_size=batch_size, shuffle=True)

			# defining trial for hyperparam sweep
			def auto_objective(trial):
				'''
				Function for Optuna to define LoL-simple or LoL's method
				'''

				model = MLP(num_feats, num_classes).to(device)
				lr = trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2, 1e-1])
				l2 = trial.suggest_categorical("l2", [0, 1e-2, 1e-1])
				optimizer = optim.Adam(model.parameters(), weight_decay=l2, lr=lr)
				
				vote_loader = wrench_loader
				batch_size = 128

				# computing coverage weights
				coverage_weights = prop_weights * batch_size
				trial_val_accs = np.zeros(num_epochs)
				trial_test_accs = np.zeros(num_epochs)

				if args.LoL:
					alpha = trial.suggest_categorical("alpha", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
					grad_val = trial.suggest_float("grad_val", 0, 5)
					x_rand = random_sample(num_rand, train_data.shape[1]).to(device)

				for i in range(num_epochs):
					
					# Training of the model
					model.train()
					for x, vote in vote_loader:
						x, vote = x.to(device), vote.to(device)
						optimizer.zero_grad()
						outputs = model(x)

						if args.LoL_simple:
							loss = LoL_simple(outputs, vote, device, coverage_weights, weights=weights)

						elif args.LoL:
							loss = LoL(model, x, vote, x_rand, lf_inds, device, coverage_weights, optimizer,
															weights=weights, num_classes=num_classes, 
															grad_val=grad_val, alpha=alpha, 
															method=grad_method)
						
						loss.backward()
						optimizer.step()

					# Validation of the model
					model.eval()
					val_preds = []
					test_preds = []
					with torch.no_grad():
						for x, _ in val_loader:
							x = x.to(device)
							ws_preds = torch.max(model(x), dim=1)[1].cpu().detach().numpy()
							val_preds.append(ws_preds)
					
						for x, _ in test_loader:
							x = x.to(device)
							auto_preds = torch.max(model(x), dim=1)[1].cpu().detach().numpy()
							test_preds.append(auto_preds)

					val_preds = np.concatenate(val_preds)
					val_accuracy = np.mean(val_preds == val_labels)
					test_preds = np.concatenate(test_preds)
					test_accuracy = np.mean(test_preds == test_labels)
					
					trial_val_accs[i] = val_accuracy
					trial_test_accs[i] = test_accuracy

					# Handle pruning based on the intermediate value every 3 epochs
					if i % 3 == 0:
						trial.report(val_accuracy, i)			
						if trial.should_prune():
							raise optuna.exceptions.TrialPruned()
				
				# getting best val & test accuracy over epochs
				best_val_accuracy, best_ind = np.max(trial_val_accs), np.argmax(trial_val_accs)
				best_test_accuracy = trial_test_accs[best_ind]

				del vote_loader
				gc.collect()
				# storing best test accuracy for later
				test_accs[trial.number] = best_test_accuracy				
				return best_val_accuracy

			sampler = TPESampler(seed = seed) 
			study = optuna.create_study(sampler=sampler, direction="maximize")
			study.optimize(auto_objective, n_trials=num_trials, timeout=50000, gc_after_trial=True)

			pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
			complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
			trial = study.best_trial

			if args.LoL_simple:
				print("Best LoL-simple Accuracy", test_accs[trial.number])
			if args.LoL:
				print("Best LoL Accuracy", test_accs[trial.number])

			res.append(test_accs[trial.number])

		del train_data, train_votes, train_labels, val_data, val_votes, val_labels, test_data, test_votes, test_labels
		gc.collect()
	
	# printing final results
	for i in res:
		print(i)


if __name__ == "__main__":
	main()
