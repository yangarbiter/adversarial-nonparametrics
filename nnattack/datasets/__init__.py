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
                random_state=var_value['random_seed'])
        
        ord = auto_var.get_var("ord")

        if auto_var.get_var("ord") == 2:                                             
            if 'mnist' in auto_var.get_variable_value("dataset"):                    
                eps = [0.1 * i for i in range(0, 41)]                                
            else:                                                                    
                eps = [0.01 * i for i in range(0, 41)]                               
        elif auto_var.get_var("ord") == 1:                                           
            eps = [0.01 * i for i in range(0, 41)]                                   
        elif auto_var.get_var("ord") == np.inf:                                      
            eps = [0.01 * i for i in range(0, 41)]                                   

        return X, y

    @register_var()
    @staticmethod
    def iris(auto_var, var_value, inter_var):
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        return X, y

    @register_var()
    @staticmethod
    def wine(auto_var, var_value, inter_var):
        from sklearn.datasets import load_wine
        X, y = load_wine(return_X_y=True)

    @register_var(argument=r"mnist17_(?P<n_samples>\d+)(?P<n_dims>_pca\d+)?")
    @staticmethod
    def mnist17(auto_var, var_value, inter_var, n_samples, n_dims):
        from keras.datasets import mnist
        from sklearn.decomposition import PCA
        n_samples = int(n_samples)
        n_dims = int(n_dims[1:]) if not n_dims else None

        (X, y), (_, _) = mnist.load_data()
        X = X.reshape(len(X), -1)
        idx1 = np.random.choice(np.where(y==1)[0], n_samples//2, replace=False)
        idx2 = np.random.choice(np.where(y==7)[0], n_samples//2, replace=False)
        y[idx1] = 0
        y[idx2] = 1
        X = np.vstack((X[idx1], X[idx2])).astype(np.float) / 255.
        y = np.concatenate((y[idx1], y[idx2]))

        if not n_dims:
            pca = PCA(n_components=n_dims, random_state=inter_var['random_state'])
            X = pca.fit_transform(X)

        return X, y

    @register_var(argument=r"mnist35_(?P<n_samples>\d+)(?P<n_dims>_pca\d+)?")
    @staticmethod
    def mnist35(auto_var, var_value, inter_var, n_samples, n_dims):
        from keras.datasets import mnist
        from sklearn.decomposition import PCA
        n_samples = int(n_samples)
        n_dims = int(n_dims[1:]) if not n_dims else None

        (X, y), (_, _) = mnist.load_data()
        X = X.reshape(len(X), -1)
        idx1 = np.random.choice(np.where(y==3)[0], n_samples//2, replace=False)
        idx2 = np.random.choice(np.where(y==5)[0], n_samples//2, replace=False)
        y[idx1] = 0
        y[idx2] = 1
        X = np.vstack((X[idx1], X[idx2])).astype(np.float) / 255.
        y = np.concatenate((y[idx1], y[idx2]))

        if not n_dims:
            pca = PCA(n_components=n_dims, random_state=inter_var['random_state'])
            X = pca.fit_transform(X)

        return X, y
        
    @register_var(argument=r"fashion_mnist06_(?P<n_samples>\d+)(?P<n_dims>_pca\d+)?")
    @staticmethod
    def fashion_mnist06(auto_var, var_value, inter_var, n_samples, n_dims):
        from keras.datasets import fashion_mnist
        from sklearn.decomposition import PCA
        n_samples = int(n_samples)
        n_dims = int(n_dims[1:]) if not n_dims else None

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

        if not n_dims:
            pca = PCA(n_components=n_dims, random_state=inter_var['random_state'])
            X = pca.fit_transform(X)

        return X, y
        
    @register_var(argument=r"fashion_mnist35_(?P<n_samples>\d+)(?P<n_dims>_pca\d+)?")
    @staticmethod
    def fashion_mnist35(auto_var, var_value, inter_var, n_samples, n_dims):
        from keras.datasets import fashion_mnist
        from sklearn.decomposition import PCA
        n_samples = int(n_samples)
        n_dims = int(n_dims[1:]) if not n_dims else None

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

        if not n_dims:
            pca = PCA(n_components=n_dims, random_state=inter_var['random_state'])
            X = pca.fit_transform(X)

        return X, y