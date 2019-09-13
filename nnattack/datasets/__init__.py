import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import PCA

from autovar.base import RegisteringChoiceType, register_var, VariableClass

LINF_EPS = [0.01 * i for i in range(0, 81, 1)]

class DatasetVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines the dataset to use"""
    var_name = 'dataset'

    @register_var(argument=r"halfmoon_(?P<n_samples>\d+)",
                  shown_name="halfmoon")
    @staticmethod
    def halfmoon(auto_var, var_value, inter_var, n_samples):
        """halfmoon dataset, n_samples gives the number of samples"""
        from sklearn.datasets import make_moons
        n_samples = int(n_samples)
        X, y = make_moons(n_samples=n_samples, noise=0.25,
                          random_state=auto_var.get_var("random_seed"))

        if auto_var.get_var("ord") == 2:
            eps = [0.01 * i for i in range(0, 41, 1)]
        elif auto_var.get_var("ord") == 1:
            eps = [0.01 * i for i in range(0, 41, 1)]
        elif auto_var.get_var("ord") == np.inf:
            eps = LINF_EPS

        return X, y, eps

    @register_var()
    @staticmethod
    def iris(auto_var, var_value, inter_var):
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        return X, y, LINF_EPS

    @register_var()
    @staticmethod
    def wine(auto_var, var_value, inter_var):
        from sklearn.datasets import load_wine
        X, y = load_wine(return_X_y=True)
        return X, y, LINF_EPS

    @register_var()
    @staticmethod
    def german(auto_var, var_value, inter_var):
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#german.numer
        X, y = load_svmlight_file("./nnattack/datasets/files/german.numer")
        X = X.todense()
        y[y==-1] = 0
        y = y.astype(int)
        return X, y, LINF_EPS

    @register_var(argument=r"splice(?P<n_dims>_pca\d+)?", shown_name="splice")
    @staticmethod
    def splice(auto_var, var_value, inter_var, n_dims):
        X, y = load_svmlight_file("./nnattack/datasets/files/splice")
        X = X.todense()
        y[y==-1] = 0
        y = y.astype(int)

        n_dims = int(n_dims[4:]) if n_dims else None
        if n_dims:
            pca = PCA(n_components=n_dims,
                    random_state=auto_var.get_var("random_seed"))
            X = pca.fit_transform(X)
        return X, y, LINF_EPS

    @register_var()
    @staticmethod
    def svmguide3(auto_var, var_value, inter_var):
        X, y = load_svmlight_file("./nnattack/datasets/files/svmguide3")
        X = X.todense()
        y[y==-1] = 0
        y = y.astype(int)
        return X, y, LINF_EPS

    @register_var()
    @staticmethod
    def diabetes(auto_var, var_value, inter_var):
        X, y = load_svmlight_file("./nnattack/datasets/files/diabetes")
        X = X.todense()
        y[y==-1] = 0
        y = y.astype(int)
        return X, y, LINF_EPS

    @register_var()
    @staticmethod
    def fourclass(auto_var, var_value, inter_var):
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/fourclass
        X, y = load_svmlight_file("./nnattack/datasets/files/fourclass")
        X = X.todense()
        y[y==-1] = 0
        y = y.astype(int)
        return X, y, LINF_EPS

    @register_var()
    @staticmethod
    def australian(auto_var, var_value, inter_var):
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian
        X, y = load_svmlight_file("./nnattack/datasets/files/australian")
        X = X.todense()
        y[y==-1] = 0
        y[y==1] = 1
        y = y.astype(int)
        return X, y, LINF_EPS

    @register_var()
    @staticmethod
    def cancer(auto_var, var_value, inter_var):
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/breast-cancer
        X, y = load_svmlight_file("./nnattack/datasets/files/breast-cancer")
        X = X.todense()
        y[y==2] = 0
        y[y==4] = 1
        y = y.astype(int)
        return X, y, LINF_EPS

    @register_var(argument=r"ijcnn1_(?P<n_samples>\d+)", shown_name="ijcnn")
    @staticmethod
    def ijcnn1(auto_var, var_value, inter_var, n_samples):
        n_samples = int(n_samples)
        X, y = load_svmlight_file("./nnattack/datasets/files/ijcnn1.tr")
        X = X.todense()

        idx1 = np.random.choice(np.where(y==1)[0], n_samples//2, replace=False)
        idx2 = np.random.choice(np.where(y==-1)[0], n_samples//2, replace=False)
        y[idx1] = 0
        y[idx2] = 1
        X = np.vstack((X[idx1], X[idx2])).astype(np.float)
        y = np.concatenate((y[idx1], y[idx2]))
        y = y.astype(int)

        return X, y, LINF_EPS

    @register_var(argument=r"covtypebin_(?P<n_samples>\d+)", shown_name="covtype")
    @staticmethod
    def covtypebin(auto_var, var_value, inter_var, n_samples):
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2
        n_samples = int(n_samples)
        X, y = load_svmlight_file("./nnattack/datasets/files/covtype.libsvm.binary")
        X = X.todense()

        idx1 = np.random.choice(np.where(y==1)[0], n_samples//2, replace=False)
        idx2 = np.random.choice(np.where(y==2)[0], n_samples//2, replace=False)
        y[idx1] = 0
        y[idx2] = 1
        X = np.vstack((X[idx1], X[idx2])).astype(np.float)
        y = np.concatenate((y[idx1], y[idx2]))
        y = y.astype(int)

        return X, y, LINF_EPS

    @register_var(argument=r"covtype_(?P<n_samples>\d+)")
    @staticmethod
    def covtype(auto_var, var_value, inter_var, n_samples):
        n_samples = int(n_samples)

        # https://archive.ics.uci.edu/ml/datasets/covertype
        data = np.genfromtxt('./nnattack/datasets/files/covtype.data', delimiter=',')
        ty = data[:, -1].astype(int) - 1
        tX = data[:, :-1].astype(float)

        idx = np.random.choice(np.arange(len(tX)), n_samples, replace=False)
        X, y = tX[idx], ty[idx]
        #X, y = np.zeros((n_samples, tX.shape[1])), np.zeros(n_samples)
        #for i in range(7):
        #    idx = np.random.choice(np.where(ty==i)[0], n_samples//7, replace=False)
        #    X[(n_samples//7)*i:(n_samples//7)*(i+1)] = tX[idx]
        #    y[(n_samples//7)*i:(n_samples//7)*(i+1)] = ty[idx]
        #X, y = X[:(n_samples//7)*7], y[:(n_samples//7)*7]

        return X, y, LINF_EPS

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

        return X, y, LINF_EPS

    @register_var(argument=r"digits(?P<n_dims>_pca\d+)?")
    @staticmethod
    def digits(auto_var, var_value, inter_var, n_dims):
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)

        n_dims = int(n_dims[4:]) if n_dims else None
        if n_dims:
            pca = PCA(n_components=n_dims,
                    random_state=auto_var.get_var("random_seed"))
            X = pca.fit_transform(X)

        return X, y, LINF_EPS

    @register_var(argument=r"fullmnist(?P<n_dims>_pca\d+)?", shown_name="mnist")
    @staticmethod
    def mnist(auto_var, inter_var, n_dims):
        from keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
        if auto_var.get_var("ord") == 2:
            eps = [0.1 * i for i in range(0, 41, 1)]
        else:
            eps = LINF_EPS

        n_dims = int(n_dims[4:]) if n_dims else None
        if n_dims:
            x_train = x_train.reshape((len(x_train), -1))
            x_test = x_test.reshape((len(x_test), -1))
            pca = PCA(n_components=n_dims,
                    random_state=auto_var.get_var("random_seed"))
            X = np.vstack((x_train, x_test))
            X = pca.fit_transform(X)
            x_train, x_test = X[:len(x_train)], X[len(x_train):]

        return x_train, y_train, x_test, y_test, eps

    @register_var(argument=r"fullfashion(?P<n_dims>_pca\d+)?", shown_name="fashion")
    @staticmethod
    def fashion(auto_var, inter_var, n_dims):
        from keras.datasets import fashion_mnist

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
        if auto_var.get_var("ord") == 2:
            eps = [0.1 * i for i in range(0, 41, 1)]
        else:
            eps = LINF_EPS

        n_dims = int(n_dims[4:]) if n_dims else None
        if n_dims:
            x_train = x_train.reshape((len(x_train), -1))
            x_test = x_test.reshape((len(x_test), -1))
            pca = PCA(n_components=n_dims,
                    random_state=auto_var.get_var("random_seed"))
            X = np.vstack((x_train, x_test))
            X = pca.fit_transform(X)
            x_train, x_test = X[:len(x_train)], X[len(x_train):]

        return x_train, y_train, x_test, y_test, eps

    @register_var(argument=r"mnist17_(?P<n_samples>\d+)(?P<n_dims>_pca\d+)?",
                  shown_name="mnist17")
    @staticmethod
    def mnist17(auto_var, var_value, inter_var, n_samples, n_dims):
        from keras.datasets import mnist
        from sklearn.decomposition import PCA
        n_samples = int(n_samples)
        n_dims = int(n_dims[4:]) if n_dims else None

        (X, y), (_, _) = mnist.load_data()
        X = X.reshape(len(X), -1)
        idx1 = np.random.choice(np.where(y==1)[0], n_samples//2, replace=False)
        idx2 = np.random.choice(np.where(y==7)[0], n_samples//2, replace=False)
        y[idx1] = 0
        y[idx2] = 1
        X = np.vstack((X[idx1], X[idx2])).astype(np.float) / 255.
        y = np.concatenate((y[idx1], y[idx2]))

        if n_dims:
            pca = PCA(n_components=n_dims,
                    random_state=auto_var.get_var("random_seed"))
            X = pca.fit_transform(X)

        if auto_var.get_var("ord") == 2:
            eps = [0.1 * i for i in range(0, 41, 1)]
        else:
            eps = LINF_EPS

        return X, y, eps

    @register_var(argument=r"mnist35_(?P<n_samples>\d+)(?P<n_dims>_pca\d+)?",
                  shown_name="mnist35")
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
            eps = [0.1 * i for i in range(0, 41, 1)]
        else:
            eps = LINF_EPS

        return X, y, eps

    @register_var(argument=r"fashion_mnist06_(?P<n_samples>\d+)(?P<n_dims>_pca\d+)?",
                  shown_name="f-mnist06")
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
            eps = [0.1 * i for i in range(0, 41, 1)]
        else:
            eps = LINF_EPS

        return X, y, eps

    @register_var(argument=r"fashion_mnist35_(?P<n_samples>\d+)(?P<n_dims>_pca\d+)?",
                  shown_name="f-mnist35")
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
            eps = [0.1 * i for i in range(0, 41, 1)]
        else:
            eps = LINF_EPS

        return X, y, eps
