from docplex.mp.model import Model
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

import numpy as np

from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_moons

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_dataset(x, y, feat0=0, feat1=1):
    colors = ['b.', 'r.', 'g.', 'k.', 'c.', 'm.']
    class_labels = np.unique(y).astype(int)
    for k in class_labels:
        plt.plot(x[y == k, feat0], x[y == k, feat1], colors[k % 7])


def plot_decision_regions(x, y, classifier, resolution=1e-3):
    # setup marker generator and color map
    colors = ('blue', 'red', 'lightgreen', 'black', 'cyan', 'magenta')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = x[:, 0].min() - 0.02, x[:, 0].max() + 0.02
    x2_min, x2_max = x[:, 1].min() - 0.02, x[:, 1].max() + 0.02
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())


def make_gaussian_datasets(n, mu, sigma):
    '''Generating a d-dimensional Gaussian dataset with k classes.'''
    n = np.array(n)  # cast to ndarray
    mu = np.array(mu)
    sigma = np.array(sigma)

    n_classes = mu.shape[0]  # n.size or n.shape[0]
    n_features = mu.shape[1]
    n_samples = np.sum(n)

    X = np.zeros(shape=(n_samples, n_features))
    Y = np.zeros(shape=(n_samples,))

    start_idx = 0
    for k in range(n_classes):  # loop over classes
        z = np.random.randn(n[k], n_features)  # sampling from N(0,1)
        xk = z * sigma[k, :] + mu[k, :]  # transform z to sample from N(mu,sigma)
        yk = k * np.ones(shape=(n[k],))  # generate nk labels equal to k
        X[start_idx:start_idx + n[k], :] = xk
        Y[start_idx:start_idx + n[k]] = yk
        start_idx += n[k]
    return X, Y


class SecSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, lambd=1):
        self.lambd = lambd

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        num_samples = X.shape[0]
        num_features = X.shape[1]

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        t = np.zeros(shape=(y.size, 1))
        t[:, 0] = 2 * y - 1  # rescale labels from {0,1} to {-1,+1}

        # Form SVM with Linf regularization problem.
        m = Model(name='Linf regularizer SVM')
        m.init_numpy()
        # w = cp.Variable((num_features, 1))
        w = m.continuous_var_matrix(num_features, 1, name="w")
        # b = cp.Variable()
        b = m.continuous_var(name="b")
        # loss = cp.sum(cp.pos(1 - cp.multiply(t, X @ w + b)))
        f = (sum(X[i, j] * w[j, 0] for j in range(X.shape[1]) + b) for i in range(X.shape[0]) )
        loss = sum(2)
        # reg = cp.norm(w, 'inf')
        #
        # lambd = cp.Parameter(nonneg=True)
        # lambd.value = self.lambd  # set lambd stored in self
        #
        # prob = cp.Problem(cp.Minimize(loss / num_samples + lambd * reg))
        # prob.solve()
        #
        # self.w = w.value
        # self.b = b.value

        # Return the classifier
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        scores = X @ self.w + self.b
        ypred = scores.flatten() > 0
        return ypred

def main():
    # x, y = make_moons(n_samples=1000, noise=0.2)
    n = [1000, 1000]  # number of points in each class
    mu = [[-5, -5], [+5, +5]]
    sigma = [[3, 3], [3, 3]]

    x, y = make_gaussian_datasets(n, mu, sigma)

    splitter = ShuffleSplit(n_splits=1, random_state=0, train_size=0.5)

    scaler = MinMaxScaler()
    # clf = svm.SVC(kernel='rbf', C=10, gamma=10.0)
    # clf_name = 'SVM RBF'
    clf = SecSVM(lambd=0.1)
    clf_name = 'SecSVM'

    for tr_idx, ts_idx in splitter.split(x, y):
        xtr = x[tr_idx, :]
        ytr = y[tr_idx]
        xts = x[ts_idx, :]
        yts = y[ts_idx]

        xtr = scaler.fit_transform(xtr)
        xts = scaler.transform(xts)

        clf.fit(xtr, ytr)
        ypred = clf.predict(xts)
        error = (ypred != yts).mean()
        print("Test error: {:.1%}".format(error))


    # plot the last classifier
    plt.figure(figsize=(5,5))
    plt.title(clf_name)
    plot_decision_regions(xtr, ytr, clf)
    plot_dataset(xtr, ytr)
    plt.text(0.05, 0.05, "Test error: {:.1%}".format(error),
            bbox=dict(facecolor='white'))
    plt.show()



    print("")


def simple2():
    cost = np.random.randint(1, 10, (4, 4))

    assignment_model = Model('Assignment')

    x = assignment_model.binary_var_matrix(cost.shape[0],
                                           cost.shape[1],
                                           name="x")

    assignment_model.add_constraints((sum(x[i, j] for i in range(cost.shape[0])) <= 1
                                    for j in range(cost.shape[1])),
                                    names='work_load')

    assignment_model.add_constraints((sum(x[i, j] for j in range(cost.shape[1])) == 1
                                    for i in range(cost.shape[0])),
                                    names='task_completion')

    obj = sum(cost[i, j] * x[i, j] for i in range(cost.shape[0]) for j in range(cost.shape[1]))
    assignment_model.set_objective('min', obj)

    assignment_model.print_information()
    print(assignment_model.export_as_lp_string())
    assignment_model.solve()
    assignment_model.print_solution()



    print("")




if __name__ == '__main__':
    main()




    print("")

