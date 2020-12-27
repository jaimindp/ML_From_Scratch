import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of h_t for t=0,...,T-1
		self.betas = []       # list of beta_t for t=0,...,T-1
		self.errors = []
		return

	@abstractmethod

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		########################################################
		# TODO
		########################################################
		total_preds = np.zeros((len(features)))
		for idx in range(len(self.clfs_picked)):
			preds = self.clfs_picked[idx].predict(features)
			total_preds += np.inner(self.betas[idx], preds)
		total_preds = np.sign(total_preds).tolist()

		return total_preds


class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO
		############################################################
		# init. 

		label_arr = np.array(labels)

		D = np.array([1/len(features) for i in range(len(labels))])
		for i in range(self.T):
			
			epsilon_t = float('inf')
			model = None
			best_preds = None
			for j in self.clfs:
				pred = np.array(j.predict(features))
				epsilon = np.dot(D,np.array(pred != label_arr).astype(int))
				if epsilon < epsilon_t:
					epsilon_t = epsilon
					model = j
					best_preds = pred

			beta_t = 0.5 * np.log((1-epsilon_t)/epsilon_t)
			sign = np.array(best_preds != label_arr).astype(int)
			sign = np.array(list(map(lambda x : x + (x-1), sign)))

			D *= np.exp(beta_t * sign)
			D /= np.sum(D)

			self.clfs_picked.append(model)
			self.betas.append(beta_t)

		return

		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)

