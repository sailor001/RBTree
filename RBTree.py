import numpy as np
#########
class SplitQuestion(object):
	"""docstring for SplitQuestion"""
	def __init__(self, attrID=0, point_or_category=0):
		super(SplitQuestion, self).__init__()
		self.attrID = attrID
		self.point_or_category = point_or_category
		# for category attributes: x[attrID] == point_or_category
		# for continuous attributes: x[attrID] <= point_or_category

	# we only consider continuous attributes for simplicity
	def test_forOneInstance(self, x):
		return x[self.attrID] <= self.point_or_category

	def test(self, X):
		return X[:, self.attrID] <= self.point_or_category

class RBNode(object):
	"""docstring for RBNode"""
	def __init__(self, depth, split, additive_func, sample_ids, X, Y):
		super(RBNode, self).__init__()
		self.sample_ids = sample_ids
		self.split = split
		self.depth = depth
		self.X = X
		self.Y = Y
		self.additive_func = additive_func
		# self.class_distribution = softmax(self.additive_func)
		# additive_func denotes F_t
		self.is_leaf = True
		# after grow_stump, set the node as an internal node

	def find_best_split(self, class_num):
		# g, h can try at F- G/H rather than at F
		score = float('-inf')
		split_attr = 0
		split_point = 0
		###############
		sample_num = np.sum(self.sample_ids)
		class_distribution = softmax(self.additive_func)
		print(self.additive_func, class_distribution)
		print(compute_class_distribution(self.Y[self.sample_ids], class_num, np.sum(self.sample_ids)))
		############
		g_array = np.zeros((sample_num, class_num))
		g_array[np.arange(sample_num), self.Y[self.sample_ids]] = 1
		g_array += np.tile(class_distribution, (sample_num, 1))
		h_array = np.tile(class_distribution - class_distribution ** 2, (sample_num, 1))
		G = np.sum(g_array, axis=0)
		H = np.sum(h_array, axis=0)
		print('G : ', G, 'H : ', H)
		print('h[0] : ', h_array[0])
		##############
		X = self.X[self.sample_ids, :]
		for i in range(X.shape[1]):
			G_L = np.zeros(class_num)
			H_L = np.zeros(class_num)
			sorted_ids = np.argsort(X[:, i])
			for j in sorted_ids[:-1]:
				G_L += g_array[j]
				H_L += h_array[j]
				G_R = G - G_L
				H_R = H - H_L
				if any(H_L * G_L * H) == 0:
					print(i, j)
					print(H_L, H_R)
					assert False
				score_new = max(score, np.sum(G_L*G_L/H_L + G_R*G_R/H_R - G*G/H))
				if score_new > score :
					score = score_new
					split_attr = i
					split_point = X[j, i]
		##############
		L_ids = X[:, split_attr] <= split_point
		G_L = np.sum(g_array[L_ids, :], axis=0)
		H_L = np.sum(h_array[L_ids, :], axis=0)
		if any(G_L == G) or any(H_L == H):
			print(X[:, split_attr])
			assert False
		func_L = self.additive_func - G/H - G_L / H_L
		func_R = self.additive_func - G/H - (G - G_L) / (H - H_L)

		self.split.attrID = split_attr
		self.split.point_or_category = split_point

		return func_L, func_R

	def grow_stump(self, func_L, func_R):
		L_sample_ids = self.sample_ids.copy()
		L_sample_ids[L_sample_ids] = self.split.test(self.X[self.sample_ids])
		R_sample_ids = np.bitwise_xor(self.sample_ids, L_sample_ids)
		LChild = RBNode(self.depth + 1, SplitQuestion(), func_L, L_sample_ids, self.X, self.Y)
		RChild = RBNode(self.depth + 1, SplitQuestion(), func_R, R_sample_ids, self.X, self.Y)
		self.LChild = LChild
		self.RChild = RChild


class RBTreeClassifier(object):
	"""docstring for RBTreeClassifier"""
	def __init__(self, max_depth=float('inf'), min_samples_split=2):
		super(RBTreeClassifier, self).__init__()
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split

	def fit(self, X, Y):
		self.X = X
		self.Y = Y
		self.classNum = self.Y.max() + 1
		self.sampleNum = self.X.shape[0]
		##########
		func0 = np.log(compute_class_distribution(self.Y, self.classNum, self.sampleNum))
		###########
		self.root_node = RBNode(1, SplitQuestion(), func0, np.ones((self.sampleNum,), dtype=bool), self.X, self.Y)
		self.leaf_num = 1
		self.tree_depth = self.bulid_subtree(self.root_node)

	def bulid_subtree(self, node):
		# stop conditions
		is_leaf = node.depth >= self.max_depth or \
		np.linalg.norm(softmax(node.additive_func)) == 1 or \
		node.sample_ids.sum() < self.min_samples_split or \
		is_same_atrributes(self.X[node.sample_ids, :]) or \
		is_all_equal(self.Y[node.sample_ids])

		if is_leaf :
			# node.is_leaf = True
			return node.depth

		func_L, func_R = node.find_best_split(self.classNum)
		node.grow_stump(func_L, func_R)
		node.is_leaf = False
		self.leaf_num += 1
		L_subtree_depth = self.bulid_subtree(node.LChild)
		R_subtree_depth = self.bulid_subtree(node.RChild)
		return max(L_subtree_depth, R_subtree_depth)		

	def predict_forOneInstance(self, x):
		present_node = self.root_node
		while not(present_node.is_leaf) : 
			if present_node.split.test_forOneInstance(x) :
				present_node = present_node.LChild
			else:
				present_node = present_node.RChild
		return np.argmax(present_node.additive_func)

	def predict(self, X):
		m = X.shape[0]
		Y_predicted = np.zeros((m,), dtype=int)
		for i in range(m):
			x = X[i]
			Y_predicted[i] = self.predict_forOneInstance(x)
		return Y_predicted

################
def softmax(x):
	exp_x = np.exp(x)
	return exp_x / np.sum(exp_x)

def is_all_equal(x):
	x_min, x_max = x.min(), x.max()
	return (x_min == x_max)

def is_same_atrributes(X):
	max_array = np.max(X, axis=0)
	min_array = np.min(X, axis=0)
	return all(max_array == min_array)

def compute_class_distribution(Y, class_num, sample_num):
	ratio_each_class = [sum(Y == k) / sample_num for k in range(class_num)]
	return np.array(ratio_each_class)		