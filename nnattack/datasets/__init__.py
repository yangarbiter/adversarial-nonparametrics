import numpy as np
from sklearn.model_selection import train_test_split

from autovar.base import RegisteringChoiceType, register_var, VariableClass

class DatasetVarClass(VariableClass, metaclass=RegisteringChoiceType):
    var_name = 'dataset'

    @register_var(argument=r"halfmoon_(?P<n_samples>\d+)")
    @staticmethod
    def halfmoon(auto_var, var_value, inter_var, n_samples):
        from sklearn.datasets import make_moons
        n_samples = int(n_samples)
        X, y = make_moons(n_samples=n_samples, noise=0.25,
                          random_state=auto_var.get_var("random_seed"))

        if auto_var.get_var("ord") == 2:
            eps = [0.01 * i for i in range(0, 41, 5)]
        elif auto_var.get_var("ord") == 1:
            eps = [0.01 * i for i in range(0, 41, 5)]
        elif auto_var.get_var("ord") == np.inf:
            eps = [0.01 * i for i in range(0, 41, 5)]

        return X, y, eps

    @register_var()
    @staticmethod
    def iris(auto_var, var_value, inter_var):
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        eps = [0.01 * i for i in range(0, 41, 5)]
        return X, y, eps

    @register_var()
    @staticmethod
    def wine(auto_var, var_value, inter_var):
        from sklearn.datasets import load_wine
        X, y = load_wine(return_X_y=True)
        eps = [0.01 * i for i in range(0, 41, 5)]
        return X, y, eps

    @register_var()
    @staticmethod
    def abalone(auto_var, var_value, inter_var):
        # http://archive.ics.uci.edu/ml/datasets/Abalone
        data = np.genfromtxt('./nnattack/datasets/files/abalone.data', dtype='str', delimiter=',')
        # get female only
        data = [data[i] for i in range(len(data)) if data[i][0] == 'F']
        X = [data[i][1:8] for i in range(len(data))]
        X = np.array([list(map(float, X[i])) for i in range(len(X))])
        # half of the abalones are 11 years old and above, so the classification task is whether age >= 11
        y = np.array([1 if int(data[i][8]) >= 11 else 0 for i in range(len(data))])

        eps = [0.01 * i for i in range(0, 41, 5)]

        return X, y, eps

    @register_var(argument=r"digits(?P<n_dims>_pca\d+)?")
    @staticmethod
    def digits(auto_var, var_value, inter_var, n_dims):
        from sklearn.datasets import load_digits
        from sklearn.decomposition import PCA
        X, y = load_digits(return_X_y=True)
        eps = [0.01 * i for i in range(0, 41, 5)]

        n_dims = int(n_dims[4:]) if n_dims else None
        if n_dims:
            pca = PCA(n_components=n_dims,
                    random_state=auto_var.get_var("random_seed"))
            X = pca.fit_transform(X)

        return X, y, eps

    @register_var(argument=r"mnist35_(?P<n_samples>\d+)(?P<n_dims>_pca\d+)?")
    @staticmethod
    def mnist35(auto_var, var_value, inter_var, n_samples, n_dims):
        from keras.datasets import mnist
        from sklearn.decomposition import PCA
        n_samples = int(n_samples)
        n_dims = int(n_dims[4:]) if n_dims else None

        (X, y), (_, _) = mnist.load_data()
        X = X.reshape(len(X), -1)
        idx1 = np.random.choice(np.where(y==3)[0], n_samples//2, replace=False)
        idx2 = np.random.choice(np.where(y==5)[0], n_samples//2, replace=False)
        y[idx1] = 0
        y[idx2] = 1
        X = np.vstack((X[idx1], X[idx2])).astype(np.float) / 255.
        y = np.concatenate((y[idx1], y[idx2]))

        if n_dims:
            pca = PCA(n_components=n_dims,
                    random_state=auto_var.get_var("random_seed"))
            X = pca.fit_transform(X)

        if auto_var.get_var("ord") == 2:
            eps = [0.1 * i for i in range(0, 41, 5)]
        else:
            eps = [0.01 * i for i in range(0, 41, 5)]

        return X, y, eps

    @register_var(argument=r"fashion_mnist06_(?P<n_samples>\d+)(?P<n_dims>_pca\d+)?")
    @staticmethod
    def fashion_mnist06(auto_var, var_value, inter_var, n_samples, n_dims):
        from keras.datasets import fashion_mnist
        from sklearn.decomposition import PCA
        n_samples = int(n_samples)
        n_dims = int(n_dims[4:]) if n_dims else None

        (X, y), (_, _) =  fashion_mnist.load_data()
        X = X.reshape(len(X), -1)
        idx1 = np.random.choice(np.where(y==0)[0], n_samples//2, replace=False)
        idx2 = np.random.choice(np.where(y==6)[0], n_samples//2, replace=False)
        y = np.copy(y)
        y.setflags(write=1)
        y[idx1] = 0
        y[idx2] = 1
        X = np.vstack((X[idx1], X[idx2])).astype(np.float) / 255.
        y = np.concatenate((y[idx1], y[idx2]))

        if n_dims:
            pca = PCA(n_components=n_dims,
                    random_state=auto_var.get_var("random_seed"))
            X = pca.fit_transform(X)

        if auto_var.get_var("ord") == 2:
            eps = [0.1 * i for i in range(0, 41, 5)]
        else:
            eps = [0.01 * i for i in range(0, 41, 5)]

        return X, y, eps

    @register_var(argument=r"fashion_mnist35_(?P<n_samples>\d+)(?P<n_dims>_pca\d+)?")
    @staticmethod
    def fashion_mnist35(auto_var, var_value, inter_var, n_samples, n_dims):
        from keras.datasets import fashion_mnist
        from sklearn.decomposition import PCA
        n_samples = int(n_samples)
        n_dims = int(n_dims[4:]) if n_dims else None

        (X, y), (_, _) =  fashion_mnist.load_data()
        X = X.reshape(len(X), -1)
        idx1 = np.random.choice(np.where(y==3)[0], n_samples//2, replace=False)
        idx2 = np.random.choice(np.where(y==5)[0], n_samples//2, replace=False)
        y = np.copy(y)
        y.setflags(write=1)
        y[idx1] = 0
        y[idx2] = 1
        X = np.vstack((X[idx1], X[idx2])).astype(np.float) / 255.
        y = np.concatenate((y[idx1], y[idx2]))

        if n_dims:
            pca = PCA(n_components=n_dims,
                    random_state=auto_var.get_var("random_seed"))
            X = pca.fit_transform(X)

        if auto_var.get_var("ord") == 2:
            eps = [0.1 * i for i in range(0, 41, 5)]
        else:
            eps = [0.01 * i for i in range(0, 41, 5)]

        return X, y, eps
