from functools import partial

import numpy as np

from autovar.base import RegisteringChoiceType, VariableClass, register_var

class AttackVarClass(VariableClass, metaclass=RegisteringChoiceType):
    var_name = "attack"

    @register_var()
    @register_var(argument=r"nnopt_k(?P<n_neighbors>\d+)(?P<n_search>_\d+)")
    @staticmethod
    def nnopt(auto_var, var_value, inter_var, n_neighbors, n_search):
        from .nn_attack import NNAttack
        n_neighbors = int(n_neighbors)
        n_search = int(n_search)
        return NNAttack(inter_var['trnX'], inter_var['trny'],
            n_neighbors=n_neighbors, farthest=n_search,
            transformer=inter_var['transformer'], ord=auto_var.get_var('ord'))

    @register_var()
    @register_var(argument=r"rev_nnopt_k(?P<n_neighbors>\d+)_(?P<n_search>\d+)_region")
    @staticmethod
    def rev_nnopt_region(auto_var, var_value, inter_var, n_neighbors, n_search):
        from .nn_attack import RevNNAttack
        n_neighbors = int(n_neighbors)
        n_search = int(n_search)
        return RevNNAttack(inter_var['trnX'], inter_var['trny'],
            farthest=n_search,
            method='region',
            n_neighbors=n_neighbors,
            transformer=inter_var['transformer'],
            ord=auto_var.get_var('ord'))

    @register_var(argument=r"rev_nnopt_k(?P<n_neighbors>\d+)_(?P<n_search>\d+)")
    @staticmethod
    def rev_nnopt(auto_var, var_value, inter_var, n_neighbors, n_search):
        from .nn_attack import RevNNAttack
        n_neighbors = int(n_neighbors)
        n_search = int(n_search)

        return RevNNAttack(inter_var['trnX'], inter_var['trny'],
            farthest=n_search,
            n_neighbors=n_neighbors,
            transformer=inter_var['transformer'],
            ord=auto_var.get_var('ord'))

    @register_var()
    @staticmethod
    def nnopt_k1_all(auto_var, var_value, inter_var):
        from .nn_attack import NNAttack
        n_neighbors = 1
        return NNAttack(inter_var['trnX'], inter_var['trny'],
            n_neighbors=n_neighbors, farthest=-1,
            transformer=inter_var['transformer'], ord=auto_var.get_var('ord'))
    
    @register_var()
    @staticmethod
    def kernelSubTf_1_pgd(auto_var, var_value, inter_var):
        from .kernel_sub_tf import KernelSubTf
        attack_model = KernelSubTf(
            sess=inter_var['sess'],
            attack='pgd',
            ord=auto_var.get_var('ord'),
            c=0.1,
            transformer=inter_var['transformer'],
        )
        return attack_model

    @register_var(argument=r"direct_k(?P<n_neighbors>\d+)")
    @staticmethod
    def direct(auto_var, var_value, inter_var, n_neighbors):
        from .direct import DirectAttack
        trnX = inter_var['trnX']
        trny = inter_var['trny']
        ord = auto_var.get_var('ord')
        n_neighbors = int(n_neighbors)

        attack_model = DirectAttack(n_neighbors=n_neighbors, ord=ord)
        attack_model.fit(trnX, trny)
        return attack_model

    @register_var()
    @staticmethod
    def kernel_sub_pgd(auto_var, var_value, inter_var):
        return inter_var['model']

    @register_var()
    @staticmethod
    def fgsm(auto_var, var_value, inter_var):
        from .fgsm import KerasFGSM
        trnX = inter_var['trnX']
        trny = inter_var['trny']
        return KerasFGSM(
            transformer=inter_var['transformer'],
            sess=inter_var['sess'],
            model=inter_var['model'].model,
            n_feats=trnX.shape[1],
            n_classes=len(np.unique(trny)),
            lbl_enc=inter_var['lbl_enc'],
            ord=auto_var.get_var('ord'),
        )

    @register_var()
    @staticmethod
    def pgd(auto_var, var_value, inter_var):
        return inter_var['model']

    @register_var()
    @staticmethod
    def dt_attack_opt(auto_var, var_value, inter_var):
        from .dt_attack import DTAttack
        attack_model = DTAttack(
            clf=inter_var['tree_clf'],
            ord=auto_var.get_var('ord'),
            transformer=inter_var['transformer'],
            random_state=inter_var['random_state'],
        )
        return attack_model

    @register_var()
    @staticmethod
    def sklinsvc_opt(auto_var, var_value, inter_var):
        return inter_var['model']

    @register_var()
    @staticmethod
    def sklr_opt(auto_var, var_value, inter_var):
        return inter_var['model']

    @register_var()
    @staticmethod
    def skada_opt(auto_var, var_value, inter_var):
        from .ada_attack import ADAAttack
        attack_model = ADAAttack(
            clf=inter_var['tree_clf'],
            ord=auto_var.get_var('ord'),
            transformer=inter_var['transformer'],
            n_features=inter_var['trnX'].shape[1],
            random_state=inter_var['random_state'],
        )
        return attack_model
