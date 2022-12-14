import sys
sys.path.append("../")

import numpy as np
import torch
import torch.utils.data as data
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
import pickle

from wrench.dataset.dataset import TextDataset

def get_lf_inds(dataset, feat_inds, count_vec):

	'''
	Script to get gradient index information for NLP tasks -> the weak labelers information are copied from the WRENCH benchmark
	and compute which indices the weak labelers look at via the input count_vec used to preprocess the data

	Args:
	dataset - the dataset (youtube, yelp, agnews, chemprot, or imdb)
	feat_inds - list of the vocabulary of the input text dataset
	count_vec - sklearn count vectorizer used to preprocess the dataset 
	'''

	if dataset == "youtube":
		youtube_inds = [
			{1: [feat_inds.index("my")]}, #keyword_my
			{1: [feat_inds.index("subscribe")]}, #keyword_subscribe
			{1: [feat_inds.index("link")]}, #keyword_link
			{1: [feat_inds.index("please"), feat_inds.index("plz")]}, #keyword_please
			{0: [feat_inds.index("song")]}, #keyword_song
			{1: [feat_inds.index("check"), feat_inds.index("out")]}, #regex_check_out
			[], #short_comment - no gradient info so not used in LoL gradient computation
			[], #has_person_nlp - no gradient info so not used in LoL gradient computation
			[], #textblob_polarity - no gradient info so not used in LoL gradient computation
			[], #textblob_subjectivity - no gradient info so not used in LoL gradient computation
		]

		return youtube_inds

	elif dataset == "yelp":

		gen_pos = ["outstanding", "perfect", "great", "good", "nice", "best", "excellent", "worthy", "awesome", "enjoy", "positive", "pleasant", "wonderful", "amazing"]  
		gen_neg = ["bad", "worst", "horrible", "awful", "terrible", "nasty", "shit", "distasteful", "dreadful", "negative"]
		mood_pos = ["happy", "pleased", "delighted", "contented", "glad", "thankful", "satisfied"]
		mood_neg = ["sad", "annoy", "disappointed", "frustrated", "upset", "irritated", "harassed", "angry", "pissed"]
		service_pos = ["friendly", "patient", "considerate", "enthusiastic", "attentive", "thoughtful", "kind", "caring", "helpful", "polite", "efficient", "prompt"]
		service_neg = ["slow", "offended", "rude", "indifferent", "arrogant"]
		price_pos = ["cheap", "reasonable", "inexpensive", "economical"]
		price_neg = ["overpriced", "expensive", "costly", "high-priced"]
		env_pos = ["clean", "neat", "quiet", "comfortable", "convenien", "tidy", "orderly", "cosy", "homely"]
		env_neg = ["noisy", "mess", "chaos", "dirty", "foul"]
		food_pos = ["tasty", "yummy", "delicious", "appetizing", "good-tasting", "delectable", "savoury", "luscious", "palatable"]
		food_neg = ["disgusting", "gross", "insipid"]

		POS = 1
		NEG = 0
		# creating indices to care about for gradients / note filtering to remove keywords that arent in train corpus
		yelp_inds = [
			[], #textblob_lf
			{POS: [feat_inds.index("recommend")]}, #keyword_recommend
			{POS: count_vec.transform(gen_pos).nonzero()[1],
			 NEG: count_vec.transform(gen_neg).nonzero()[1]}, #keyword_general
			{POS: count_vec.transform(mood_pos).nonzero()[1],
			 NEG: count_vec.transform(mood_neg).nonzero()[1]}, #keyword_mood
			{POS: count_vec.transform(service_pos).nonzero()[1],
			 NEG: count_vec.transform(service_neg).nonzero()[1]}, #service
			{POS: count_vec.transform(price_pos).nonzero()[1],
			 NEG: count_vec.transform(price_neg).nonzero()[1]}, #keyword_price
			{POS: count_vec.transform(env_pos).nonzero()[1],
			 NEG: count_vec.transform(env_neg).nonzero()[1]}, #keyword_environemnt
			{POS: count_vec.transform(food_pos).nonzero()[1],
			 NEG: count_vec.transform(food_neg).nonzero()[1]}, #keyword_food
		]

		return yelp_inds

	elif dataset == "agnews":

		## LF1  0: world
		r1 = ["atomic", "captives", "baghdad", "israeli", "iraqis", "iranian", "afghanistan", "wounding", "terrorism", "soldiers", \
		"palestinians", "palestinian", "policemen", "iraqi", "terrorist", 'north korea', 'korea', \
		'israel', 'u.n.', 'egypt', 'iran', 'iraq', 'nato', 'armed', 'peace']

		## LF2  0: world
		r2= [' war ', 'prime minister', 'president', 'commander', 'minister',  'annan', "military", "militant", "kill", 'operator']

		## LF3  1: sports
		r3 = ["goals", "bledsoe", "coaches",  "touchdowns", "kansas", "rankings", "no.", \
			"champ", "cricketers", "hockey", "champions", "quarterback", 'club', 'team',  'baseball', 'basketball', 'soccer', 'football', 'boxing',  'swimming', \
			'world cup', 'nba',"olympics","final", "finals", 'fifa',  'racist', 'racism'] 

		## LF4   1: sports
		r4 = ['athlete',  'striker', 'defender', 'goalkeeper',  'midfielder', 'shooting guard', 'power forward', 'point guard', 'pitcher', 'catcher', 'first base', 'second base', 'third base','shortstop','fielder']

		## LF5   1: sports
		r5=['lakers','chelsea', 'piston','cavaliers', 'rockets', 'clippers','ronaldo', \
			'celtics', 'hawks','76ers', 'raptors', 'pacers', 'suns', 'warriors','blazers','knicks','timberwolves', 'hornets', 'wizards', 'nuggets', 'mavericks', 'grizzlies', 'spurs', \
			'cowboys', 'redskins', 'falcons', 'panthers', 'eagles', 'saints', 'buccaneers', '49ers', 'cardinals', 'texans', 'seahawks', 'vikings', 'patriots', 'colts', 'jaguars', 'raiders', 'chargers', 'bengals', 'steelers', 'browns', \
			'braves','marlins','mets','phillies','cubs','brewers','cardinals', 'diamondbacks','rockies', 'dodgers', 'padres', 'orioles', 'sox', 'yankees', 'jays', 'sox', 'indians', 'tigers', 'royals', 'twins','astros', 'angels', 'athletics', 'mariners', 'rangers', \
			'arsenal', 'burnley', 'newcastle', 'leicester', 'manchester united', 'everton', 'southampton', 'hotspur','tottenham', 'fulham', 'watford', 'sheffield','crystal palace', 'derby', 'charlton', 'aston villa', 'blackburn', 'west ham', 'birmingham city', 'middlesbrough', \
			'real madrid', 'barcelona', 'villarreal', 'valencia', 'betis', 'espanyol','levante', 'sevilla', 'juventus', 'inter milan', 'ac milan', 'as roma', 'benfica', 'porto', 'getafe', 'bayern', 'schalke', 'bremen', 'lyon', 'paris saint', 'monaco', 'dynamo']

		## LF6  3: tech
		r6 = ["technology", "engineering", "science", "research", "cpu", "windows", "unix", "system", 'computing',  'compute']#, "wireless","chip", "pc", ]

		## LF7  3: tech
		r7= ["google", "apple", "microsoft", "nasa", "yahoo", "intel", "dell", \
			'huawei',"ibm", "siemens", "nokia", "samsung", 'panasonic', \
			't-mobile', 'nvidia', 'adobe', 'salesforce', 'linkedin', 'silicon', 'wiki'
		]

		## LF8  - 2:business
		r8= ["stock", "account", "financ", "goods", "retail", 'economy', 'chairman', 'bank', 'deposit', 'economic', 'dow jones', 'index', '$',  'percent', 'interest rate', 'growth', 'profit', 'tax', 'loan',  'credit', 'invest']

		## LF9  - 2:business
		r9= ["delta", "cola", "toyota", "costco", "gucci", 'citibank', 'airlines']


		agnews_inds = [
			{0: count_vec.transform(r1).nonzero()[1]},
			{0: count_vec.transform(r2).nonzero()[1]},
			{1: count_vec.transform(r3).nonzero()[1]},
			{1: count_vec.transform(r4).nonzero()[1]},
			{1: count_vec.transform(r5).nonzero()[1]},
			{3: count_vec.transform(r6).nonzero()[1]},
			{3: count_vec.transform(r7).nonzero()[1]},
			{2: count_vec.transform(r8).nonzero()[1]},
			{2: count_vec.transform(r9).nonzero()[1]},
		]

		return agnews_inds
	
	elif dataset == "chemprot":

		# Labels
		# "0": "Part of",
		# "1": "Regulator",
		# "2": "Upregulator",
		# "3": "Downregulator",
		# "4": "Agonist",
		# "5": "Antagonist",
		# "6": "Modulator",
		# "7": "Cofactor",
		# "8": "Substrate/Product",
		# "9": "NOT"

		lf1 = ["amino acid"] # lab 1
		lf2 = ["replace"] # lab 1
		lf3 = ["mutant", "mutat"] # lab 1
		lf4 = ["bind"] # lab 2
		lf5 = ["interact"] # lab 2
		lf6 = ["affinit"] # lab 2
		lf7 = ["activat"] # lab 3
		lf8 = ["icreas"] # lab 3
		lf9 = ["induc"] # lab 3
		lf10 = ["stimulat"] # lab 3
		lf11 = ["upregulat"] # lab 3
		lf12 = ["downregulat", "down-regulat"] # lab 4
		lf13 = ["reduc"] # lab 4
		lf14 = ["inhibit"] # lab 4
		lf15 = ["decreas"] # lab 4
		lf16 = ["agnoi", "\tagnoi"] # lab 5
		lf17 = ["antagon"] # lab 6
		lf18 = ["modulat"] # lab 7
		lf19 = ["allosteric"] # lab 7
		lf20 = ["cofactor"] # lab 8
		lf21 = ["substrate"] # lab 9
		lf22 = ["transport"] # lab 9
		lf23 = ["catalyz", "catalys"] # lab 9
		lf24 = ["produc"] # lab 9
		lf25 = ["conver"] # lab 9
		lf26 = ["not"] # lab 10 (0?)

		vocab_dict = count_vec.vocabulary_

		chemprot_inds = [
			{0: count_vec.transform(lf1).nonzero()[1]},
			{0: count_vec.transform(lf2).nonzero()[1]},
			{0: count_vec.transform(lf3).nonzero()[1]},
			{1: count_vec.transform(lf4).nonzero()[1]},
			{1: count_vec.transform(lf5).nonzero()[1]},
			{1: count_vec.transform(lf6).nonzero()[1]},
			{2: count_vec.transform(lf7).nonzero()[1]},
			{2: count_vec.transform(lf8).nonzero()[1]},
			{2: count_vec.transform(lf9).nonzero()[1]},
			{2: count_vec.transform(lf10).nonzero()[1]},
			{2: count_vec.transform(lf11).nonzero()[1]},
			{3: count_vec.transform(lf12).nonzero()[1]},
			{3: count_vec.transform(lf13).nonzero()[1]},
			{3: count_vec.transform(lf14).nonzero()[1]},
			{3: count_vec.transform(lf15).nonzero()[1]},
			{4: count_vec.transform(lf16).nonzero()[1]},
			{5: count_vec.transform(lf17).nonzero()[1]},
			{6: count_vec.transform(lf18).nonzero()[1]},
			{6: count_vec.transform(lf19).nonzero()[1]},
			{7: count_vec.transform(lf20).nonzero()[1]},
			{8: count_vec.transform(lf21).nonzero()[1]},
			{8: count_vec.transform(lf22).nonzero()[1]},
			{8: count_vec.transform(lf23).nonzero()[1]},
			{8: count_vec.transform(lf24).nonzero()[1]},
			{8: count_vec.transform(lf25).nonzero()[1]},
			{9: count_vec.transform(lf26).nonzero()[1]},
		]

		return chemprot_inds

	elif dataset == "imdb":
		# Labels 
		# "0": "Negative",
		# "1": "Positive"
		pre_neg = ["no ", "not ", "never ", "n t ", "zero ", "0 ", "tiny ", "little ", "less", "rare", "worse"]
		pre_pos=["will ", " ll ", "would ", " d ", "can t wait to "]
		expression=[" next time", " again", " rewatch", " anymore", " rewind"] # since used in both ways -> gradient is zero

		lf2_neg = [" than this", " than the film", " than the movie"]
		lf2_pos = []

		lf3_neg = ["bad", "worst", "horrible", "awful", "terrible", "crap", "shit", "garbage", "rubbish", "waste"]
		lf3_pos = ["masterpiece", "outstanding", "perfect", "great", "good", "nice", "best", "excellent", "worthy", "awesome", "enjoy", "positive", "pleasant", "wonderful", "amazing", "superb", "fantastic", "marvellous", "fabulous"]

		lf4_neg = ["fast forward", "n t finish"]
		lf4_pos = []

		lf5_neg = ["to sleep", "fell asleep", "boring", "dull", "plain"]
		lf5_pos = ["well written", "absorbing", "attractive", "innovative", "instructive", "interesting", "touching", "moving"]

		imdb_inds = [
			{0: count_vec.transform(pre_neg).nonzero()[1],
			 1: count_vec.transform(pre_pos).nonzero()[1]},
			{0: count_vec.transform(lf2_neg).nonzero()[1],
			 1: count_vec.transform(lf2_pos).nonzero()[1]},
			{0: count_vec.transform(lf3_neg).nonzero()[1],
			 1: count_vec.transform(lf3_pos).nonzero()[1]},
			{0: count_vec.transform(lf4_neg).nonzero()[1],
			 1: count_vec.transform(lf4_pos).nonzero()[1]},
			{0: count_vec.transform(lf5_neg).nonzero()[1],
			 1: count_vec.transform(lf5_pos).nonzero()[1]},			
		]

		return imdb_inds

def create_dataloader(x_data, labels, batch_size=200, shuffle=True):

	'''
	Function to generate a pytorch dataloader from data and labels

	Args:
	x_data - input data x for the dataloader
	labels - labels for teh dataloader
	'''

	return data.DataLoader(data.TensorDataset(torch.tensor(x_data, dtype=torch.float), 
											  torch.tensor(labels, dtype=torch.long)), 
											  batch_size=batch_size, shuffle=shuffle)

def create_grad_dataloader(x_data, grad, labels, batch_size=200, shuffle=True):

	'''
	Function to generate a pytorch dataloader from data and labels

	Args:
	x_data - input data x for the dataloader
	labels - labels for teh dataloader
	'''

	return data.DataLoader(data.TensorDataset(torch.tensor(x_data, dtype=torch.float),
											  torch.tensor(grad, dtype=torch.float),
											  torch.tensor(labels, dtype=torch.long)), 
											  batch_size=batch_size, shuffle=shuffle)

def get_data_from_wrench(dataset="youtube", embs=None):

	'''
	Script to load various train, val, and test datasets for experiments from the WRENCH benchmark
	(only to be run from this file)
	'''

	## Loading Data
	dataset_path = dataset

	if embs == None:

		train_dataset = TextDataset(path=dataset_path, split='train')
		valid_dataset = TextDataset(path=dataset_path, split='valid')
		test_dataset = TextDataset(path=dataset_path, split='test')

		train_data_l = [x["text"] for x in train_dataset.examples]
		val_data_l = [x["text"] for x in valid_dataset.examples]
		test_data_l = [x["text"] for x in test_dataset.examples]

		cv_func, count_vec = train_dataset.extract_feature(extract_fn='bow', return_extractor=True)
		valid_dataset.extract_feature(extract_fn=cv_func, return_extractor=False)
		test_dataset.extract_feature(extract_fn=cv_func, return_extractor=False)
		
		train_data, train_votes, train_labels = train_dataset.features, train_dataset.weak_labels, train_dataset.labels
		val_data, val_votes, val_labels = valid_dataset.features, valid_dataset.weak_labels, valid_dataset.labels
		test_data, test_votes, test_labels = test_dataset.features, test_dataset.weak_labels, test_dataset.labels

		train_data, train_votes, train_labels = np.array(train_data), np.array(train_votes), np.array(train_labels)
		val_data, val_votes, val_labels = np.array(val_data), np.array(val_votes), np.array(val_labels)
		test_data, test_votes, test_labels = np.array(test_data), np.array(test_votes), np.array(test_labels)

		feat_inds = count_vec.get_feature_names()
		lf_inds = get_lf_inds(dataset, feat_inds, count_vec)
		return (train_data, train_votes, train_labels), (val_data, val_votes, val_labels), (test_data, test_votes, test_labels), lf_inds

	else: # using pre-trained model embeddings (get feature extractor to compute gradients as well)

		print("Extracting embeddings")

		train_dataset = TextDataset(path=dataset_path, split='train', extract_fn='bert_grad')
		valid_dataset = TextDataset(path=dataset_path, split='valid', extract_fn='bert')
		test_dataset = TextDataset(path=dataset_path, split='test', extract_fn='bert')

		train_data, train_votes, train_labels = train_dataset.features, train_dataset.weak_labels, train_dataset.labels
		val_data, val_votes, val_labels = valid_dataset.features, valid_dataset.weak_labels, valid_dataset.labels
		test_data, test_votes, test_labels = test_dataset.features, test_dataset.weak_labels, test_dataset.labels

		train_data, train_votes, train_labels = np.array(train_data), np.array(train_votes), np.array(train_labels)
		val_data, val_votes, val_labels = np.array(val_data), np.array(val_votes), np.array(val_labels)
		test_data, test_votes, test_labels = np.array(test_data), np.array(test_votes), np.array(test_labels)

		# getting grad info
		train_grad = np.array(train_dataset.grad)
		return (train_data, train_votes, train_labels), (val_data, val_votes, val_labels), (test_data, test_votes, test_labels), train_grad


def save_wrench_splits(dataset):

	'''
	Script to save all data from Wrench splits (only to be ran from this file)
	'''

	(train_data, train_votes, train_labels), (val_data, val_votes, val_labels), \
		(test_data, test_votes, test_labels), lf_inds = get_data_from_wrench(dataset)

	np.save(dataset + "/wrench/train_data", train_data)
	np.save(dataset + "/wrench/train_votes", train_votes)
	np.save(dataset + "/wrench/train_labels", train_labels)
	np.save(dataset + "/wrench/val_data", val_data)
	np.save(dataset + "/wrench/val_votes", val_votes)
	np.save(dataset + "/wrench/val_labels", val_labels)
	np.save(dataset + "/wrench/test_data", test_data)
	np.save(dataset + "/wrench/test_votes", test_votes)
	np.save(dataset + "/wrench/test_labels", test_labels)
	pickle.dump(lf_inds, open(dataset + "/inds.p", "wb"))

def save_all_data(dataset):

	'''
	Script to save all data for a given dataset (only to be ran from this file)
	'''

	(train_data, train_votes, train_labels), (val_data, val_votes, val_labels), \
		(test_data, test_votes, test_labels), lf_inds = get_data_from_wrench(dataset)

	all_data = np.concatenate([train_data, val_data, test_data])
	votes = np.concatenate([train_votes, val_votes, test_votes])
	labels = np.concatenate([train_labels, val_labels, test_labels])

	np.save(dataset + "/data", all_data)
	np.save(dataset + "/votes", votes)
	np.save(dataset + "/labels", labels)
	pickle.dump(lf_inds, open(dataset + "/inds.p", "wb"))


def load_all_data(dataset):
	'''
	Function to load all data (not split into train, val, test)
	'''
	
	path = "./data/all_data/" + dataset + "_"
	all_data, votes, labels = np.load(path + "data.npy"), np.load(path + "votes.npy"), np.load(path + "labels.npy")
	lf_inds = pickle.load(open("./data/" + dataset + "/inds.p", "rb"))
	return all_data, votes, labels, lf_inds

def gen_split(all_data, votes, labels, num_val, num_class, seed):
	'''
	Function to split data and get train, val, test splits given a random seed
	'''

	# split to 20% test data and 10% val data
	train_data, test_data, train_votes, test_votes, train_labels, test_labels = train_test_split(all_data, votes, labels, test_size=0.2, random_state=seed)
	train_data, val_data, train_votes, val_votes, train_labels, val_labels = train_test_split(train_data, train_votes, train_labels, test_size=num_val * num_class, random_state=seed)
	return (train_data, train_votes, train_labels), (val_data, val_votes, val_labels), \
		(test_data, test_votes, test_labels)

def save_split(d_path, dataset, votes, labels, num_val, num_class, seed):
	(train_data, train_votes, train_labels), (val_data, val_votes, val_labels), \
		(test_data, test_votes, test_labels) = gen_split(dataset, votes, labels, num_val, num_class, seed)
	
	print("Train", train_data.shape)
	print("Val", val_data.shape)
	print("Test", test_data.shape)

	np.save(dataset + "/wrench/train_data", train_data)
	np.save(dataset + "/wrench/train_votes", train_votes)
	np.save(dataset + "/wrench/train_labels", train_labels)
	np.save(dataset + "/wrench/val_data", val_data)
	np.save(dataset + "/wrench/val_votes", val_votes)
	np.save(dataset + "/wrench/val_labels", val_labels)
	np.save(dataset + "/wrench/test_data", test_data)
	np.save(dataset + "/wrench/test_votes", test_votes)
	np.save(dataset + "/wrench/test_labels", test_labels)

def load_split_data(dataset, seed, num_val):

	'''
	Function to load a particular split of data (must be run \emph{after} generating and saving data first)

	Args:
	dataset - which particular dataset
	seed - which seed of the split
	num_val - size of validation split
	'''

	if num_val == "wrench":
		train_path = "./data/" + dataset + "/wrench/train_"
		val_path = "./data/" + dataset + "/wrench/val_"
		test_path = "./data/" + dataset + "/wrench/test_"
	else:
		train_path = "./data/" + dataset + "/" + str(num_val) + "/seed" + str(seed) +  "/_train_"
		val_path = "./data/"   + dataset + "/" + str(num_val) + "/seed" + str(seed) +  "/_val_"
		test_path = "./data/"  + dataset + "/" + str(num_val) + "/seed" + str(seed) +  "/_test_"

	train_data, train_votes, train_labels = np.load(train_path + "data.npy"), np.load(train_path + "votes.npy"), np.load(train_path + "labels.npy")
	val_data, val_votes, val_labels = np.load(val_path + "data.npy"), np.load(val_path + "votes.npy"), np.load(val_path + "labels.npy")
	test_data, test_votes, test_labels = np.load(test_path + "data.npy"), np.load(test_path + "votes.npy"), np.load(test_path + "labels.npy")
	lf_inds = pickle.load(open("./data/" + dataset + "/inds.p", "rb"))

	return (train_data, train_votes, train_labels), (val_data, val_votes, val_labels), \
		(test_data, test_votes, test_labels), lf_inds

def load_split_awa2_data(seed):
	'''
	Function to load AwA2 data
	'''

	train_path = "./AWA2/" + "seed_" + str(seed) + "_train_"
	val_path = "./AWA2/" + "seed_" + str(seed) + "_val_"
	test_path = "./AWA2/" + "seed_" + str(seed) + "_test_"

	train_data, train_votes, train_labels = torch.load(train_path + "data"), torch.load(train_path + "votes"), torch.load(train_path + "labels")
	val_data, val_votes, val_labels = torch.load(val_path + "data"), torch.load(val_path + "votes"), torch.load(val_path + "labels")
	test_data, test_votes, test_labels = torch.load(test_path + "data"), torch.load(test_path + "votes"), torch.load(test_path + "labels")
	train_grad, val_grad = torch.load(train_path + "grad"), torch.load(val_path + "grad")
	
	return (train_data, train_labels, train_votes, train_grad), \
		(val_data, val_labels, val_votes, val_grad), \
			(test_data, test_labels, _, _)


if __name__ == "__main__":

	datasets = ["youtube", "chemprot"]
	# datasets = ["youtube", "yelp", "chemprot", "imdb", "agnews"]
	num_val = 10 # change this to include the number of desired validation data per class
	num_class_dict = {"youtube":2, "yelp":2, "awa2":2, "agnews":4, "chemprot":10, "imdb":2}
	seeds = [0, 1, 2, 3, 4]

	# this saves all of the data into a single file to later be used for splits
	for d in datasets:
		# (train_data, train_votes, train_labels), (val_data, val_votes, val_labels), \
			# (test_data, test_votes, test_labels), lf_inds = get_data_from_wrench(d)
		
		save_wrench_splits(d)
