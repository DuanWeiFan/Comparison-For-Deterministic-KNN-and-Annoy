import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from celluloid import Camera

MNIST_PATH="../../train.csv"

class TreeNode():
    def __init__(self, digit):
        self.digit = digit
        self.left = None
        self.right = None
        self.points = []
        self.labels = []
        self.is_leave = False
        self.coef = []
        self.a, self.b = 0, 0

    def get_side(self, point):
        if np.dot(self.coef, np.append(point, [1])) > 0:
            return 0
        return 1

    def size(self):
        return len(self.points)

    def add_point(self, point):
        self.points.append(point)

    def add_label(self, label):
        self.labels.append(label)


def hashing(node):
    points = node.points
    labels = node.labels
    
    idx_list = [idx for idx in range(len(points))]

    idx1, idx2 = np.random.choice(idx_list, 2, replace=False)
    p1, p2 = points[idx1], points[idx2]

    _mid = (p1 + p2) / 2
    _n = p2 - p1

    _c = -1 * np.dot(_n, _mid)

    node.coef = np.append(_n, _c)

    left_node = TreeNode(0)
    right_node = TreeNode(1)

    node.left = left_node
    node.right = right_node


    for i in range(len(points)):
    # for point in points:
        if node.get_side(points[i]) == 0:
            left_node.add_point(points[i])
            left_node.add_label(labels[i])
        else:
            right_node.add_point(points[i])
            right_node.add_label(labels[i])

    return left_node, right_node

def query(root, point):
    root_points = root.points
    curr = root

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    camera = Camera(fig)

    if len(root.points[0]) > 3:
        # pca
        pca = PCA(n_components=3)
        pca.fit(root.points)
        pca.singular_values_



    while curr.is_leave is False:
        # ax.scatter([p[0] for p in root_points], [p[1] for p in root_points], [p[2] for p in root_points], marker='.', c='black')
        # ax.scatter([p[0] for p in curr.points], [p[1] for p in curr.points], [p[2] for p in curr.points], marker='o', c='blue')
        # ax.scatter(point[0], point[1], point[2], marker='x', c='red')
        if curr.get_side(point) == 0:
            curr = curr.left
        else:
            curr = curr.right
        # camera.snap()
    # animation = camera.animate(interval=1000, repeat=False)
    # animation.save('animation.gif', writer='PillowWriter')
    return curr.points, curr.labels

def predict(points):
    return statistics.mode(points)

def plot(points, target, neighbors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([p[0] for p in points], [p[1] for p in points], [p[2] for p in points], marker='.')
    ax.scatter(target[0], target[1], target[2], marker='x')
    ax.scatter([p[0] for p in neighbors], [p[1] for p in neighbors], [p[2] for p in neighbors], marker='o')
    plt.show()

def read_mnist():
    global MNIST_PATH
    data = pd.read_csv(MNIST_PATH)
    data = data.head(1000)
    y = data['label'].tolist()
    del data['label']
    X = data.values
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    k = 5
    total_num_of_tree = 5
    num_points = 100

    # generate 2 dimensional points
    # rand_points = np.random.rand(num_points, 3)

    # rand_labels = np.random.randint(0, 3, num_points)

    # iris = datasets.load_iris()

    # X = iris.data[:, :]
    # y = iris.target


    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train, X_test, y_train, y_test = read_mnist()

    roots = []

    for _ in range(total_num_of_tree):
        root = TreeNode(-1)

        points = []
        for i in range(len(X_train)):
            p = X_train[i]
            label = y_train[i]
            points.append(p)
            root.add_point(p)
            root.add_label(label)

        queue = [root]

        level = 0
        # BFS
        while queue:
            level += 1
            size = len(queue)
            # print(f'level: {level}, size of queue: {size}')

            for _ in range(size):
                node = queue[0]
                del queue[0]
                left_node, right_node = hashing(node)
                if left_node.size() > k:
                    queue.append(left_node)
                else:
                    left_node.is_leave = True

                if right_node.size() > k:
                    queue.append(right_node)
                else:
                    right_node.is_leave = True
        roots.append(root)

    total, correct = 0, 0
    for i in range(len(X_test)):
        point = X_test[i]
        label = y_test[i]

        labels = []
        # loop through the forest
        for root in roots:
            _neighbor, _labels = query(root, point)
            labels.extend(_labels)
        try:
            prediction = statistics.mode(labels)
        except:
            prediction = labels[0]
        total += 1
        if prediction == label:
            correct += 1

    print(f'total count: {total}, correct count: {correct}')
    print(f'accuracy: {correct / total}')


    # target = np.array([0.1, 0.3, 0.5])

    # neighbors, labels = query(root, target)
    # print(f'prediction: {statistics.mode(labels)}')
    # print(f'target: {(target[0], target[1], target[2])}, neighbors: {[(p[0], p[1], p[2]) for p in neighbors]}')
    # plot(points, target, neighbors)

if __name__ == '__main__':
    main()
