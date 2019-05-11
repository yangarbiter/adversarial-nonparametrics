from functools import partial
from copy import deepcopy

import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid, KFold
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.utils_keras import KerasModelWrapper

from autovar import AutoVar
from autovar.base import RegisteringChoiceType, VariableClass, register_var


def exp_fn(auto_var, trnX, valX, trny, valy, eps:float):
    auto_var.set_intermidiate_variable('sess', tf.Session())
    auto_var.set_intermidiate_variable("trnX", trnX)
    auto_var.set_intermidiate_variable("trny", trny)
    model = auto_var.get_var('model')
    model.fit(trnX, trny)
    auto_var.set_intermidiate_variable('model', model)
    attack = auto_var.get_var('attack')

    pert = attack.perturb(valX, valy, eps=eps)

    return (model.predict(valX + pert) == valy).mean()

def cross_validation(auto_var: AutoVar, grid, valid_eps:float):
    X = auto_var.get_intermidiate_variable('trnX')
    y = auto_var.get_intermidiate_variable('trny')

    # TODO add clone method
    sess = auto_var.inter_var.pop('sess')
    val_auto_var = deepcopy(auto_var)
    auto_var.inter_var['sess'] = sess
    val_auto_var._read_only = False
    val_auto_var._no_hooks = True

    # shuffle
    kf = KFold(n_splits=3)
    rets = []
    for i, (train, test) in enumerate(kf.split(X)):
        print(f"{i}-th cross validation.....")
        fn = partial(exp_fn, trnX=X[train], trny=y[train],
                     valX=X[test], valy=y[test], eps=valid_eps)

        params, results = val_auto_var.run_grid_params(fn, grid_params=grid)
        #params, ret = auto_var.run_grid_params(fn, grid)
        rets.append(results)
    scores = np.array(rets).mean(axis=0)
    print("xvalidation results:", params, scores)
    return params[np.argmax(scores)]


class ModelVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines which classifier to use.
    The defense is implemented in this option."""
    var_name = "model"

    @register_var(argument=r"random_forest_(?P<n_trees>\d+)(?P<depth>_d\d+)?")
    @staticmethod
    def random_forest(auto_var, var_value, inter_var, n_trees, depth):
        """Random Forest Classifier"""
        from sklearn.ensemble import RandomForestClassifier
        depth = int(depth[2:]) if depth else None
        n_trees = int(n_trees)

        model = RandomForestClassifier(
            n_estimators=n_trees,
            criterion='entropy',
            max_depth=depth,
            random_state=auto_var.get_var("random_seed"),
        )
        auto_var.set_intermidiate_variable("tree_clf", model)
        return model

    @register_var(argument=r"(?P<train>[a-zA-Z0-9]+_)?rf_(?P<n_trees>\d+)_(?P<eps>\d+)(?P<depth>_d\d+)?")
    @staticmethod
    def adv_robustrf(auto_var, var_value, inter_var, train, eps, n_trees, depth):
        """Random Forest Classifier with defense
        n_trees: number of trees in the forest
        depth: maximum depth of each tree
        train:
          None: undefended decision tree
          adv: adversarial training
          robust: robust splitting
          robustv1: adversarial pruning
          advPruning: Wang's defense for 1-NN
        eps: defense strength """
        from .adversarial_dt import AdversarialRf
        from sklearn.ensemble import RandomForestClassifier
        eps = int(eps) * 0.01
        train = train[:-1] if train else None
        depth = int(depth[2:]) if depth else None
        n_trees = int(n_trees)

        attack_model = None
        if train == 'adv':
            attack_model = auto_var.get_var("attack")
        model = AdversarialRf(
            n_estimators=n_trees,
            criterion='entropy',
            train_type=train,
            attack_model=attack_model,
            ord=auto_var.get_var("ord"),
            eps=eps,
            max_depth=depth,
            random_state=auto_var.get_var("random_seed"),
        )
        auto_var.set_intermidiate_variable("tree_clf", model)
        return model

    @register_var(argument=r"decision_tree(?P<depth>_d\d+)?")
    @staticmethod
    def decision_tree(auto_var, var_value, inter_var, depth):
        """Original Decision Tree Classifier"""
        from sklearn.tree import DecisionTreeClassifier
        depth = int(depth[2:]) if depth else None
        model = DecisionTreeClassifier(
                max_depth=depth,
                criterion='entropy',
                random_state=auto_var.get_var("random_seed")
            )
        auto_var.set_intermidiate_variable("tree_clf", model)
        return model

    @register_var(argument=r'(?P<train>[a-zA-Z0-9]+_)?decision_tree(?P<depth>_d\d+)?_(?P<eps>\d+)')
    @staticmethod
    def adv_decision_tree(auto_var, var_value, inter_var, train, eps, depth):
        """Decision Tree classifier
        train:
          None: undefended decision tree
          adv: adversarial training
          robust: robust splitting
          robustv1: adversarial pruning
          advPruning: Wang's defense for 1-NN
        eps: defense strength """
        from .adversarial_dt import AdversarialDt
        eps = int(eps) * 0.01
        train = train[:-1] if train else None
        depth = int(depth[2:]) if depth else None

        trnX, trny = inter_var['trnX'], inter_var['trny']
        model = auto_var.get_var_with_argument("model", "decision_tree")
        model.fit(trnX, trny)
        auto_var.set_intermidiate_variable("tree_clf", model)
        attack_model = None
        if train == 'adv':
            attack_model = auto_var.get_var("attack")
        model = AdversarialDt(
            eps=eps,
            criterion='entropy',
            train_type=train,
            max_depth=depth,
            attack_model=attack_model,
            ord=auto_var.get_var("ord"),
            random_state=auto_var.get_var("random_seed"))
        auto_var.set_intermidiate_variable("tree_clf", model)
        return model

    @register_var(argument=r'(?P<train>[a-zA-Z0-9]+)_kernel_sub_tf_c(?P<c>\d+)_(?P<eps>\d+)')
    @staticmethod
    def adv_kernel_sub_tf(auto_var, var_value, inter_var, train, eps, c):
        from .kernel_sub_tf import KernelSubTFModel
        c = float(c) * 0.1
        eps = float(eps) * 0.1

        clf = KernelSubTFModel(
            c=c,
            lbl_enc=inter_var['lbl_enc'],
            sess=inter_var['sess'],
            train_type=train,
            eps=eps,
            ord=auto_var.get_var("ord"),
        )
        return clf

    @register_var(argument=r'(?P<train>[a-zA-Z0-9]+_)?kernel_sub_tf_xvalid_(?P<eps>\d+)')
    @staticmethod
    def adv_kernel_xvalid(auto_var, var_value, inter_var, train, eps):
        grid = {
            'model': [
                'kernel_sub_tf_c1_1',
                'kernel_sub_tf_c1_5',
                'kernel_sub_tf_c1_10',
                'kernel_sub_tf_c1_15',
                'kernel_sub_tf_c1_20',
            ]
        }
        valid_eps = float(eps) * 0.1
        if train is not None:
            for i in range(len(grid['model'])):
                grid['model'][i] = train + grid['model'][i]
        model_name = cross_validation(auto_var, grid, valid_eps)['model']
        #auto_var.set_variable_value('model', 'model_name')
        model = auto_var.get_var_with_argument('model', model_name)
        auto_var.set_intermidiate_variable("tree_clf", model)
        return model

    @register_var(argument=r"robust1nn")
    @staticmethod
    def robust1nn(auto_var, var_value, inter_var):
        from .robust_nn import Robust_1NN
        return Robust_1NN(Delta=0.45, delta=0.1, ord=auto_var.get_var("ord"))

    @register_var(argument=r"(?P<train>[a-zA-Z0-9]+_)?nn_k(?P<n_neighbors>\d+)_(?P<eps>\d+)")
    @staticmethod
    def adv_robustnn(auto_var, var_value, inter_var, n_neighbors, train, eps):
        """ Nearest Neighbor classifier
        train:
          None: undefended
          adv: adversarial training
          robustv1: adversarial pruning
          advPruning: Wang's defense for 1-NN
        eps: defense strength """
        from .adversarial_knn import AdversarialKnn
        eps = int(eps) * 0.01
        train = train[:-1] if train else None
        n_neighbors = int(n_neighbors)

        attack_model = None
        if train == 'adv':
            attack_model = auto_var.get_var("attack")

        return AdversarialKnn(
            n_neighbors=n_neighbors,
            train_type=train,
            attack_model=attack_model,
            ord=auto_var.get_var("ord"),
            eps=eps,
        )

    @register_var(argument='knn(?P<n_neighbors>\d+)')
    @staticmethod
    def knn(auto_var, var_value, inter_var, n_neighbors):
        """Original Nearest Neighbor classifier"""
        n_neighbors = int(n_neighbors)
        return KNeighborsClassifier(n_neighbors=n_neighbors)

    @register_var()
    @staticmethod
    def kernel_sub_tf(auto_var, var_value, inter_var):
        from .kernel_sub_tf import KernelSubTFModel
        clf = KernelSubTFModel(
            c=0.1,
            lbl_enc=inter_var['lbl_enc'],
            sess=inter_var['sess'],
            ord=auto_var.get_var("ord"),
        )
        return clf

    @register_var(argument='(?P<train>[a-zA-Z0-9]+_)?sklr(?P<eps>_\d+)?')
    @staticmethod
    def sklr(auto_var, var_value, inter_var, train, eps):
        from .sklr import SkLr
        eps = float(eps[1:])*0.01 if eps else 0.
        train = train[:-1] if train else None
        clf = SkLr(
            ord=auto_var.get_var("ord"),
            train_type=train,
            eps=eps,
            solver="liblinear",
        )
        return clf

    @register_var(argument='(?P<train>[a-zA-Z0-9]+_)?adadt_(?P<n_estimators>\d+)(?P<eps>_\d+)?')
    @staticmethod
    def skadadt(auto_var, var_value, inter_var, train, n_estimators, eps):
        from .adversarial_adaboost import AdversarialAda
        from sklearn.tree import DecisionTreeClassifier
        eps = int(eps) * 0.1 if eps is not None else 0.
        n_estimators = int(n_estimators)
        train = train[1:] if train is not None else None

        if train == 'adv':
            # TODO
            attack_model = None
        else:
            attack_model = None
        model = AdversarialAda(
            base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=1),
            algorithm='SAMME',
            n_estimators=n_estimators,
            train_type=train,
            attack_model=attack_model,
            eps=eps,
            ord=auto_var.get_var("ord"),
            random_state=auto_var.get_var("random_seed"))
        auto_var.set_intermidiate_variable("tree_clf", model)
        return model

    @register_var(argument='(?P<train>[a-zA-Z0-9]+_)?sklinsvc(?P<eps>_\d+)?')
    @staticmethod
    def sklinsvc(auto_var, var_value, inter_var, train, eps):
        from .sklinsvc import SkLinSVC
        eps = float(eps[1:])*0.01 if eps else 0.
        train = train[:-1] if train else None
        clf = SkLinSVC(
            ord=auto_var.get_var("ord"),
            train_type=train,
            eps=eps,
        )
        return clf

    @register_var(argument='(?P<train>[a-zA-Z0-9]+_)?mlp(?P<eps>_\d+)?')
    @staticmethod
    def adv_mlp(auto_var, var_value, inter_var, train, eps):
        """ Multi-layer perceptrum classifier
        train:
          None: undefended
          adv: adversarial training
          robustv1: adversarial pruning
        eps: defense strength """
        from .keras_model import KerasModel
        eps = float(eps[1:])*0.01 if eps else 0.
        train = train[:-1] if train else None

        n_features = inter_var['trnX'].shape[1:]
        n_classes = len(set(inter_var['trny']))

        model = KerasModel(
            lbl_enc=inter_var['lbl_enc'],
            n_features=n_features,
            n_classes=n_classes,
            sess=inter_var['sess'],
            architecture='mlp',
            train_type=train,
            ord=auto_var.get_var("ord"),
            eps=eps,
            attacker=None,
            eps_list=inter_var['eps_list'],
            epochs=2000,
        )
        return model

    @register_var(argument='(?P<train>[a-zA-Z0-9]+_)?logistic_regression(?P<eps>_\d+)?')
    @staticmethod
    def adv_logistic_regression(auto_var, var_value, inter_var, train, eps):
        from .keras_model import KerasModel
        eps = float(eps[1:])*0.01 if eps else 0.
        train = train[:-1] if train else None

        n_features = inter_var['trnX'].shape[1:]
        n_classes = len(set(inter_var['trny']))

        model = KerasModel(
            lbl_enc=inter_var['lbl_enc'],
            n_features=n_features,
            n_classes=n_classes,
            sess=inter_var['sess'],
            architecture='logistic_regression',
            train_type=train,
            ord=auto_var.get_var("ord"),
            eps=eps,
            attacker=None,
            eps_list=inter_var['eps_list'],
            epochs=2000,
        )
        return model

    @register_var(argument='(?P<train>[a-zA-Z0-9]+_)?decision_tree_xvalid_(?P<eps>\d+)')
    @staticmethod
    def adv_dt_xvalid(auto_var, var_value, inter_var, train, eps):
        grid = {
            'model': [
                'decision_tree_1',
                'decision_tree_5',
                'decision_tree_10',
                'decision_tree_15',
                'decision_tree_20',
            ]
        }
        valid_eps = float(eps) * 0.1
        if train is not None:
            for i in range(len(grid['model'])):
                grid['model'][i] = train + grid['model'][i]
        model_name = cross_validation(auto_var, grid, valid_eps)['model']
        #auto_var.set_variable_value('model', 'model_name')
        model = auto_var.get_var_with_argument('model', model_name)
        auto_var.set_intermidiate_variable("tree_clf", model)
        return model

    #@register_var(argument=r"adv_robustv1nn_k(?P<n_neighbors>\d+)_xvalid")
    #@staticmethod
    #def adv_robustv1nn_xvalid(auto_var, var_value, inter_var, n_neighbors):
    #    n_neighbors = int(n_neighbors)
    #    grid = {
    #        'model': [
    #            f'adv_robustv1nn_k{n_neighbors}_1',
    #            f'adv_robustv1nn_k{n_neighbors}_5',
    #            f'adv_robustv1nn_k{n_neighbors}_10'
    #            f'adv_robustv1nn_k{n_neighbors}_15'
    #            f'adv_robustv1nn_k{n_neighbors}_20'
    #        ]
    #    }
    #    cross_validation(auto_var, grid)
