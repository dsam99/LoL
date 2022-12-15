# Losses over Labels (LoL): Weakly Supervised Learning via Direct Loss Construction

Our code implementation for the LoL project (AAAI 2023). 

## Installation

We provide the requirements file for our Anaconda environment; you can install the dependencies via:

```
pip install -r requirements.txt
```

## Data 

To replicate our experiments, you first need to download the data from the WRENCH benchmark (agnews, chemprot, imdb, yelp, youtube). These can be found at this link: 
https://drive.google.com/drive/folders/1v55IKG2JN9fMtKJWU48B_5_DcPWGnpTq

First, we split our data to create (and save) the train, val, and test splits used in our experiments. You can do this first by running the file

```
python process_data.py
```

which will generate the splits for the experiments using 10 validation data points per class. These splits will be located in the 'saved_data/seed{seed # here}/' directories (depending on the seed used to create the split). We note that the larger datasets (yelp, agnews) require a fair bit of memory.

## Running our Code

To run our experiments, you can use the command

```
python run.py --LoL --dataset youtube
```

This will run the experiments (over all 5 seeds) for the LoL method and print the results. There are multiple flags that you can pass to the command including:

- (--LoL): This will run the LoL approach
- (--LoL_simple): This will run the LoL-sw approach
- (--snorkel or --mv): This will run the Snorkel baseline and the MV baseline
- (--dataset youtube): This will run the youtube dataset (which you can change for 'agnews', 'chemprot', 'imdb', or 'yelp')

For many of these approaches, we highly recommend using a GPU as we perform hyperparameter optimization over many different trials especially on the imdb, yelp, and agnews datasets.

## Citation

Please cite the following paper if you use our work. Thanks!

Dylan Sam and J. Zico Kolter. "Losses over Labels: Weakly Supervised Learning via Direct Loss Construction." Proceedings of the AAAI Conference on Artificial Intelligence, 2023.

```
@inproceedings{sam2023,
  Author = {Sam, D. and Kolter, J. Z.},
  Title = {Losses over Labels: Weakly Supervised Learning via Direct Loss Construction},
  Booktitle = {AAAI Conference on Artificial Intelligence},
  Year = {2023}}
```
