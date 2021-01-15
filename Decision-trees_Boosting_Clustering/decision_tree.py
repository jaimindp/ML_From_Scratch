import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################


			entropy = 0
			branch_arr = np.array(branches)
			branch_sum = np.sum(branch_arr, axis=0)
			tot = np.sum(branch_arr)

			for c in range(len(branches)):
				for b in range(len(branches[c])):
					if branch_sum[b] != 0 and tot != 0 and branch_arr[c][b] != 0:
						entropy -= branch_sum[b] / tot * np.log2(branch_arr[c][b] / branch_sum[b])

			return entropy
			
 
		CE_min = 1e5
		features_np = np.array(self.features)
		if features_np.shape[1] == 0:
			self.splittable = False

		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################

			feature = np.array(self.features)[:, idx_dim]
			print(feature)
			feature_uniq = np.unique(feature).tolist()
			print(feature_uniq)
			num_branch = len(feature_uniq)
			print(num_branch)
		
			if num_branch < 2:
				continue

			branches = []
			for idx_cls in range(self.num_cls):
				branches.append([0]*num_branch)

			print(self.labels)
			for idx_element, element in enumerate(feature):
				branches[self.labels[idx_element]][feature_uniq.index(element)] += 1

			print(branches)

			# CE
			CE_branch = conditional_entropy(branches)
			if CE_branch < CE_min:
				CE_min = CE_branch
				self.dim_split = idx_dim
				self.feature_uniq_split = feature_uniq

		if CE_min == 1e5:
			self.splittable = False
			return

		############################################################
		# TODO: split the node, add child nodes
		############################################################
		# split from dim_split
		for idx_branch in range(len(feature_uniq)):
			child_features = []
			child_labels = []
			for idx_element in range(len(self.features)):
				if features_np[idx_element, self.dim_split] == feature_uniq[idx_branch]:
					new_feature_np = np.concatenate((features_np[idx_element,:self.dim_split],
						                         features_np[idx_element, self.dim_split+1:]))
					child_features.append(new_feature_np.tolist())
					child_labels.append(self.labels[idx_element])
			
			if len(child_labels) == len(self.labels):
				self.splittable = False
				return
			
			child = TreeNode(child_features, child_labels, self.num_cls)
			self.children.append(child)
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max


