
import numpy as np
from metric_learn import LMNN, NCA, LSML, LFDA, MLKR

from autovar.base import RegisteringChoiceType, VariableClass, register_var

class TransformerVarClass(VariableClass, metaclass=RegisteringChoiceType):
    var_name = "transformer"

    @register_var()
    @staticmethod
    def identity(auto_var, var_value, inter_var):
        return Identity()

    @register_var()
    @staticmethod
    def lmnn3(auto_var, var_value, inter_var):
        return LMNN(k=3)

    @register_var()
    @staticmethod
    def lmnn1(auto_var, var_value, inter_var):
        return LMNN(k=1)

    @register_var()
    @staticmethod
    def nca(auto_var, var_value, inter_var):
        return NCA()

    @register_var()
    @staticmethod
    def mlkr(auto_var, var_value, inter_var):
        return MLKR()

    @register_var()
    @staticmethod
    def pca(auto_var, var_value, inter_var):
        from .pca import PCA
        return PCA(random_state=inter_var['random_state'])

    @register_var()
    @staticmethod
    def nca_tf(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_chebyshev(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      center_type='chebyshev',
                      attack=None,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_rev_nnopt_k1_50_1(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 1
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "rev_nnopt_k1_50")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_rev_nnopt_k3_100_1(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 3
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "rev_nnopt_k3_100")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_rev_nnopt_k3_100_5(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 3
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "rev_nnopt_k3_100")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.5,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_nnopt_k1_all_both_1(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 1
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "nnopt_k1_all")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      adv_type='both',
                      n_neighbors=n_neighbors,
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_nnopt_k1_all_1(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 1
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "nnopt_k1_all")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_direct_3_1_max_volume(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 3
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "direct_3")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      center_type='max_volume',
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_direct_1_both_1(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 1
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "direct_1")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      adv_type='both',
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_direct_1_both_1_max_volume(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 1
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "direct_1")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      center_type='max_volume',
                      adv_type='both',
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_direct_3_both_1_max_volume(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 3
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "direct_3")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      center_type='max_volume',
                      adv_type='both',
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_direct_3_both_1(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 3
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "direct_3")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      attack=attack_model,
                      adv_type='both',
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_direct_3_1(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 3
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "direct_3")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_direct_3_5(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 3
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "direct_3")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.5,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_nnopt_k1_all_1_max_volume(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 1
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "nnopt_k1_all")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      center_type='max_volume',
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_rev_nnopt_k3_100_both_1_max_volume(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 3
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "rev_nnopt_k3_100")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      center_type='max_volume',
                      adv_type='both',
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_rev_nnopt_k3_100_1_max_volume(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 3
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "rev_nnopt_k3_100")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      center_type='max_volume',
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_nnopt_k1_all_both_5_max_volume(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 1
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "nnopt_k1_all")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      center_type='max_volume',
                      adv_type='both',
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.5,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_nnopt_k1_all_both_1_max_volume(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        n_neighbors = 1
        transformer = auto_var.get_var_with_argument('transformer', "identity")
        transformer.fit(inter_var['trnX'], inter_var['trny'])
        auto_var.set_intermidiate_variable("transformer", transformer)
        attack_model = auto_var.get_var_with_argument('attack', "nnopt_k1_all")
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      n_neighbors=n_neighbors,
                      center_type='max_volume',
                      adv_type='both',
                      attack=attack_model,
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_fgsm_1(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      attack='fgsm',
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_pgd_1(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      attack='pgd',
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_pgd_5(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      attack='pgd',
                      ord=auto_var.get_var('ord'),
                      eps=.5,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_pgd_1_max_volume(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      center_type='max_volume',
                      attack='pgd',
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_pgd_both_1(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      attack='pgd',
                      adv_type='both',
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_pgd_both_1_max_volume(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      center_type='max_volume',
                      attack='pgd',
                      adv_type='both',
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_pgd_5_max_volume(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      center_type='max_volume',
                      attack='pgd',
                      ord=auto_var.get_var('ord'),
                      eps=.5,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_pgd_1_chebyshev(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      center_type='chebyshev',
                      attack='pgd',
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_pgd_1_analytic(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      center_type='analytic',
                      attack='pgd',
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_pgd_1_uniform(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      init='uniform',
                      attack='pgd',
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_pgd_2(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      attack='pgd',
                      ord=auto_var.get_var('ord'),
                      eps=.2,
                      sess=inter_var['sess'])

    @register_var()
    @staticmethod
    def nca_tf_pgd_1_c_var(auto_var, var_value, inter_var):
        from .nca_tf import NCA_TF
        return NCA_TF(n_dim=inter_var['trnX'].shape[1],
                      c='var', 
                      attack='pgd',
                      ord=auto_var.get_var('ord'),
                      eps=.1,
                      sess=inter_var['sess'])

    #@register_var()
    #@staticmethod
    #def rca(auto_var, var_value, inter_var):
    #    from metric_learn import RCA_Supervised
    #    num_chunks = len(inter_var['trnX']) * 0.3
    #    return RCA_Supervised(num_chunks=num_chunks, chunk_size=2)


class Identity(object):

    def __init__(self):
        pass

    def fit(self, X, y):
        self.n_fets = np.shape(X)[1]
        return self

    def transform(self, X):
        return X

    def transformer(self):
        return np.eye(self.n_fets)
