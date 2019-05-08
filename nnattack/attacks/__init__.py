from functools import partial

import numpy as np

from autovar.base import RegisteringChoiceType, VariableClass, register_var

class AttackVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines which attack method to use."""
    var_name = "attack"

    @register_var(argument=r"nnopt_k(?P<n_neighbors>\d+)_(?P<n_search>\d+)")
    @staticmethod
    def nnopt(auto_var, var_value, inter_var, n_neighbors, n_search):
        from .nns.nn_attack import NNAttack
        n_neighbors = int(n_neighbors)
        n_search = int(n_search)
        return NNAttack(inter_var['trnX'], inter_var['trny'],
            n_neighbors=n_neighbors, farthest=n_search,
            ord=auto_var.get_var('ord'))

    @register_var(argument=r"rev_nnopt_k(?P<n_neighbors>\d+)_(?P<n_search>\d+)_region",
                  shown_name="RBA-Approx")
    @staticmethod
    def rev_nnopt_region(auto_var, var_value, inter_var, n_neighbors, n_search):
        """RBA-Approx for Nearest Neighbor"""
        from .nns.nn_attack import RevNNAttack
        n_neighbors = int(n_neighbors)
        n_search = int(n_search)
        return RevNNAttack(inter_var['trnX'], inter_var['trny'],
            farthest=n_search,
            method='region',
            n_neighbors=n_neighbors,
            ord=auto_var.get_var('ord'))

    @register_var(argument=r"hybrid_nnopt_k(?P<n_neighbors>\d+)_(?P<n_search>\d+)_(?P<rev_n_search>\d+)")
    @staticmethod
    def hybrid_nnopt(auto_var, var_value, inter_var, n_neighbors, n_search,
            rev_n_search):
        from .nns.nn_attack import HybridNNAttack
        n_neighbors = int(n_neighbors)
        n_search = int(n_search)
        rev_n_search = int(rev_n_search)

        return HybridNNAttack(inter_var['trnX'], inter_var['trny'],
            farthest=n_search,
            rev_farthest=rev_n_search,
            n_neighbors=n_neighbors,
            ord=auto_var.get_var('ord'))

    @register_var(argument=r"rev_nnopt_k(?P<n_neighbors>\d+)_(?P<n_search>\d+)")
    @staticmethod
    def rev_nnopt(auto_var, var_value, inter_var, n_neighbors, n_search):
        from .nns.nn_attack import RevNNAttack
        n_neighbors = int(n_neighbors)
        n_search = int(n_search)

        return RevNNAttack(inter_var['trnX'], inter_var['trny'],
            farthest=n_search,
            n_neighbors=n_neighbors,
            ord=auto_var.get_var('ord'))

    @register_var()
    @staticmethod
    def gradient_based(auto_var, var_value, inter_var):
        from .nns.gradient_based import GradientBased
        return GradientBased(
                   sess=auto_var.inter_var['sess'],
                   trnX=auto_var.inter_var['trnX'],
                   trny=auto_var.inter_var['trny'],
                   ord=auto_var.get_var('ord'),
               )

    @register_var(argument=r"nnopt_k(?P<n_neighbors>\d+)_all",
                  shown_name="RBA-Exact")
    @staticmethod
    def nnopt_all(auto_var, var_value, inter_var, n_neighbors):
        """RBA-Exact for nearest neighbor"""
        from .nns.nn_attack import NNAttack
        n_neighbors = int(n_neighbors)
        return NNAttack(inter_var['trnX'], inter_var['trny'],
            n_neighbors=n_neighbors, farthest=-1,
            ord=auto_var.get_var('ord'), n_jobs=8)

    @register_var(argument=r"kernelsub_c(?P<c>\d+)_(?P<attack>[a-zA-Z0-9]+)")
    @staticmethod
    def kernelSubTf(auto_var, var_value, inter_var, c, attack):
        from .kernel_sub_tf import KernelSubTf
        c = float(c) * 0.001
        attack_model = KernelSubTf(
            sess=inter_var['sess'],
            attack=attack,
            ord=auto_var.get_var('ord'),
            c=c,
        )
        return attack_model

    @register_var(argument=r"direct_k(?P<n_neighbors>\d+)")
    @staticmethod
    def direct(auto_var, var_value, inter_var, n_neighbors):
        """Direct Attack for Nearest Neighbor"""
        from .nns.direct import DirectAttack
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
    def pgd(auto_var, var_value, inter_var):
        """Projected gradient descent attack"""
        return inter_var['model']

    @register_var()
    @staticmethod
    def blackbox(auto_var, var_value, inter_var):
        """Cheng's black box attack (BBox)"""
        from .blackbox import BlackBoxAttack
        ret = BlackBoxAttack(
            model=inter_var['model'],
            ord=auto_var.get_var('ord'),
            random_state=inter_var['random_state'],
        )
        return ret

    @register_var()
    @staticmethod
    def dt_papernot(auto_var, var_value, inter_var):
        """Papernot's attack on decision tree"""
        from .trees.papernots import Papernots
        attack_model = Papernots(
            clf=inter_var['tree_clf'],
            ord=auto_var.get_var('ord'),
            random_state=inter_var['random_state'],
        )
        return attack_model

    @register_var(shown_name="RBA-Exact")
    @staticmethod
    def dt_attack_opt(auto_var, var_value, inter_var):
        """RBA-Exact for Decision Tree"""
        from .trees.dt_opt import DTOpt
        attack_model = DTOpt(
            clf=inter_var['tree_clf'],
            ord=auto_var.get_var('ord'),
            random_state=inter_var['random_state'],
        )
        return attack_model

    @register_var(argument=r"rf_attack_all", shown_name="RBA-Exact")
    @staticmethod
    def rf_attack_all(auto_var, var_value, inter_var):
        """RBA-Exact for Random Forest"""
        from .trees.rf_attack import RFAttack

        attack_model = RFAttack(
            trnX=inter_var['trnX'],
            trny=inter_var['trny'],
            n_searches=-1,
            method='all',
            clf=inter_var['tree_clf'],
            ord=auto_var.get_var('ord'),
            random_state=inter_var['random_state'],
        )
        return attack_model

    @register_var(argument=r"rf_attack_rev(?P<n_search>_\d+)?",
                  shown_name="RBA-Approx")
    @staticmethod
    def rf_attack_rev(auto_var, var_value, inter_var, n_search):
        """RBA-Approx for Random Forest"""
        from .trees.rf_attack import RFAttack
        n_search = int(n_search[1:]) if n_search is not None else -1

        attack_model = RFAttack(
            trnX=inter_var['trnX'],
            trny=inter_var['trny'],
            n_searches=n_search,
            method='rev',
            clf=inter_var['tree_clf'],
            ord=auto_var.get_var('ord'),
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
            n_features=inter_var['trnX'].shape[1],
            random_state=inter_var['random_state'],
        )
        return attack_model
