import math
import collections
from sklearn import datasets
from sklearn.model_selection import train_test_split

def euclidean_dist(p1, p2):
	dist = 0
	# O(d)
	for i1, i2 in zip(p1, p2):
		dist += (i1 - i2) ** 2
	return math.sqrt(dist)

def get_mode(src_list):
	# Space: O(#labels)
	counter = collections.defaultdict(int)
	for ele in [x[1] for x in src_list]:
		counter[ele] += 1
	majority = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0]
	return majority[0]

def deteministic_sklearn_predict(X_train, y_train, X_test, k=10):
	predictions = []
	# n: testing sample size = 1
	# m: training sample size
	# O(n)
	for sample in X_test:
		# calculate distance of all its neighbors
		dist_list = []
		# O(m)
		for point, target in zip(X_train, y_train):
			dist = euclidean_dist(sample, point) # O(d)
			dist_list.append((dist, target))
		# O(m*d)

		# O(m log m)
		nearest_k = sorted(dist_list, key=lambda x: x[0])[:k]

		# [1, 1, 1, 0, 0, 1] -> 1
		predictions.append(get_mode(nearest_k))
	# O(n*(m*d + m log m))
	return predictions

def get_accuracy(y_truth, predictions):
	correct = 0
	for t_truth, t_prediction in zip(y_truth, predictions):
		if t_truth == t_prediction:
			correct += 1
	return f'{round(correct/len(y_truth)*100, 2)}%'

def main():
	iris = datasets.load_iris()

	X = iris.data[:, :]
	y = iris.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	predictions = deteministic_sklearn_predict(X_train, y_train, X_test, k=10)

	print(f'ground truth: {y_test}')
	print(f'predictions: {predictions}')
	print('accuracy:', get_accuracy(y_test, predictions))

if __name__ == '__main__':
	main()
