
from nnattack.variables import auto_var
from utils import RobustExperiments

random_seed = list(range(1))
ATTACK_NORM = '2'

datasets = [
    'australian',
    'fourclass',
    'diabetes',
    'cancer',
    'halfmoon_2200',
    'covtypebin_1200',
    'fashion_mnist35_2200_pca25',
    'fashion_mnist06_2200_pca25',
    'mnist17_2200_pca25',
]

ds_eps = {
    'australian': 50,
    'fourclass': 25,
    'diabetes': 25,
    'cancer': 50,
    'halfmoon': 25,
    'covtype': 50,
    'f-mnist35': 75,
    'f-mnist06': 75,
    'mnist17': 75,
}

tree_datasets = [
    'australian',
    'fourclass',
    'diabetes',
    'cancer',
    'halfmoon_2200',
    'covtypebin_10200',
    'fashion_mnist35_10200_pca25',
    'fashion_mnist06_10200_pca25',
    'mnist17_10200_pca25',
]

class compare_attacks(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "compare_attacks_l2"
        grid_params = []
        grid_params.append({
            'model': ['knn1'],
            'ord': [ATTACK_NORM],
            'dataset': datasets,
            'attack': ['direct_k1', 'blackbox',
                       'kernelsub_c1000_pgd', 'RBA_Exact_KNN_k1'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': ['knn3'],
            'ord': [ATTACK_NORM],
            'dataset': datasets,
            'attack': ['direct_k3', 'blackbox',
                       'kernelsub_c1000_pgd', 'RBA_Approx_KNN_k3_50'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': ['decision_tree_d5'],
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': ['dt_papernots', 'blackbox', 'RBA_Exact_DT'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': ['random_forest_100_d5'],
            #'model': ['random_forest_300_d5'],
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': ['blackbox', 'RBA_Approx_RF_100'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class parametric_defense(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "parametric_defense_l2"
        grid_params = []
        grid_params.append({
            'model': [
                'logistic_regression',
                'adv_logistic_regression_50',
                'advPruning_logistic_regression_50',
            ],
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'mlp',
                'adv_mlp_50',
                'advPruning_mlp_50',
            ],
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class compare_defense(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "compare_defense_l2"
        def_strength = 50
        grid_params = []
        grid_params.append({
            'model': [
                'knn1',
                f'adv_nn_k1_{def_strength}',
                f'robustv2_nn_k1_{def_strength}',
                f'advPruning_nn_k1_{def_strength}'
            ],
            'ord': [ATTACK_NORM],
            'dataset': datasets,
            'attack': ['RBA_Exact_KNN_k1'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'knn3',
                f'adv_nn_k3_{def_strength}',
                f'advPruning_nn_k3_{def_strength}'
            ],
            'ord': [ATTACK_NORM],
            'dataset': datasets,
            'attack': ['RBA_Approx_KNN_k3_50'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'decision_tree_d5',
                f'adv_decision_tree_d5_{def_strength}',
                f'robust_decision_tree_d5_{def_strength}',
                f'advPruning_decision_tree_d5_{def_strength}',
            ],
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': ['RBA_Exact_DT'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'random_forest_100_d5',
                f'adv_rf_100_{def_strength}_d5',
                f'robust_rf_100_{def_strength}_d5',
                f'advPruning_rf_100_{def_strength}_d5',
            ],
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': ['RBA_Approx_RF_100'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'logistic_regression',
                f'adv_logistic_regression_{def_strength}',
                f'advPruning_logistic_regression_{def_strength}',
            ],
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'mlp',
                f'adv_mlp_{def_strength}',
                f'advPruning_mlp_{def_strength}',
            ],
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class tst_scores(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "tst_scores_l2"
        grid_params = []
        grid_params.append({
            'model': [
                #'knn1', 'advPruning_nn_k1_10', 'advPruning_nn_k1_30', 'advPruning_nn_k1_50',
                'knn1', 'advPruning_nn_k1_25', 'advPruning_nn_k1_50', 'advPruning_nn_k1_75',
            ],
            'ord': [ATTACK_NORM],
            'dataset': datasets,
            'attack': ['RBA_Exact_KNN_k1'], #, 'blackbox'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'knn3', 'advPruning_nn_k3_25', 'advPruning_nn_k3_50', 'advPruning_nn_k3_75',
            ],
            'ord': [ATTACK_NORM],
            'dataset': datasets,
            'attack': ['RBA_Approx_KNN_k3_50'], #, 'blackbox'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'decision_tree_d5',
                'advPruning_decision_tree_d5_25', 'advPruning_decision_tree_d5_50', 'advPruning_decision_tree_d5_75',
            ],
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': ['RBA_Exact_DT'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'random_forest_100_d5',
                'advPruning_rf_100_25_d5', 'advPruning_rf_100_50_d5', 'advPruning_rf_100_75_d5',
            ],
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': ['RBA_Approx_RF_100'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class fullds(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "fullds_l2"
        grid_params = []
        grid_params.append({
            'model': [
                'random_forest_300_d10', 'approxAP_rf_300_500_d10',
                "knn1", "approxAP_nn_k1_500",
                "knn3", "approxAP_nn_k3_500",
            ],
            'ord': [ATTACK_NORM],
            'dataset': ['fullmnist', 'fullfashion'],
            'attack': ['blackbox'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class nn_k1_robustness(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "1nn_robustness_l2"
        grid_params = []
        grid_params.append({
            'model': [
                'knn1',
                'advPruning_nn_k1_25', 'advPruning_nn_k1_50', 'advPruning_nn_k1_75',
                #'adv_nn_k1_10', 'adv_nn_k1_30', 'adv_nn_k1_50',
                #'robustv2_nn_k1_10', 'robustv2_nn_k1_30', 'robustv2_nn_k1_50',
            ],
            'ord': [ATTACK_NORM],
            'dataset': datasets,
            'attack': ['RBA_Exact_KNN_k1'],#, 'blackbox'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class nn_k3_robustness(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "3nn_robustness_l2"
        grid_params = []
        grid_params.append({
            'model': [
                'knn3',
                'advPruning_nn_k3_25', 'advPruning_nn_k3_50', 'advPruning_nn_k3_75',
                #'adv_nn_k3_10', 'adv_nn_k3_30', 'adv_nn_k3_50',
                #'robustv2_nn_k3_10', 'robustv2_nn_k3_30', 'advPruning_nn_k3_50',
            ],
            'ord': [ATTACK_NORM],
            'dataset': datasets,
            'attack': ['RBA_Approx_KNN_k3_50'],#, 'blackbox'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class rf_robustness(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "rf-robustness_l2"
        models = [
            'random_forest_100_d5',
            #'adv_rf_100_10_d5', 'adv_rf_100_30_d5', 'adv_rf_100_50_d5',
            #'robust_rf_100_10_d5', 'robust_rf_100_30_d5', 'robust_rf_100_50_d5',
            'advPruning_rf_100_25_d5', 'advPruning_rf_100_50_d5', 'advPruning_rf_100_75_d5',
            #'robustv2_rf_100_10_d5', 'robustv2_rf_100_30_d5', 'robustv2_rf_100_50_d5',
        ]
        attacks = ['RBA_Approx_RF_100']

        grid_params = []
        grid_params.append({
            'model': models,
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': attacks,
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class dt_robustness(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "dt-robustness_l2"
        models = [
            'decision_tree_d5',
            #'robust_decision_tree_d5_10', 'robust_decision_tree_d5_30', 'robust_decision_tree_d5_50',
            'advPruning_decision_tree_d5_25', 'advPruning_decision_tree_d5_50', 'advPruning_decision_tree_d5_75',
        ]
        attacks = ['RBA_Exact_DT']

        grid_params = []
        grid_params.append({
            'model': models,
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': attacks,
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class lr_ap_robustness(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "lr-ap-robustness_l2"
        models = [
            'logistic_regression',
            'advPruning_logistic_regression_25',
            'advPruning_logistic_regression_50',
            'advPruning_logistic_regression_75',
        ]
        attacks = ['pgd']

        grid_params = []
        grid_params.append({
            'model': models,
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': attacks,
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class lr_at_robustness(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "lr-at-robustness_l2"
        models = [
            'logistic_regression',
            'adv_logistic_regression_25',
            'adv_logistic_regression_50',
            'adv_logistic_regression_75',
        ]
        attacks = ['pgd']

        grid_params = []
        grid_params.append({
            'model': models,
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': attacks,
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class mlp_ap_robustness(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "mlp-ap-robustness_l2"
        models = [
            'mlp', 'advPruning_mlp_25', 'advPruning_mlp_50', 'advPruning_mlp_75',
        ]
        attacks = ['pgd']

        grid_params = []
        grid_params.append({
            'model': models,
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': attacks,
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class mlp_at_robustness(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "mlp-at-robustness_l2"
        models = [
            'mlp',
            'adv_mlp_25', 'adv_mlp_50', 'adv_mlp_75',
        ]
        attacks = ['pgd']

        grid_params = []
        grid_params.append({
            'model': models,
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': attacks,
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)


class nn1_def(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "rf-robustness_l2"
        attacks = ['RBA_Exact_KNN_k1']

        grid_params = []
        for ds in datasets:
            v, k = [ds], ds_eps[auto_var.get_var_shown_name("dataset", ds)]
            models = [
                'knn1',
                f'adv_nn_k1_{k}',
                f'robustv2_nn_k1_{k}',
                f'advPruning_nn_k1_{k}',
            ]

            grid_params.append({
                'model': models,
                'ord': [ATTACK_NORM],
                'dataset': v,
                'attack': attacks,
                'random_seed': random_seed,
            })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class nn3_def(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "rf-robustness_l2"
        attacks = ['RBA_Approx_KNN_k3_50']

        grid_params = []
        for ds in datasets:
            v, k = [ds], ds_eps[auto_var.get_var_shown_name("dataset", ds)]
            models = [
                'knn3',
                f'adv_nn_k3_{k}',
                f'robustv2_nn_k3_{k}',
                f'advPruning_nn_k3_{k}',
            ]

            grid_params.append({
                'model': models,
                'ord': [ATTACK_NORM],
                'dataset': v,
                'attack': attacks,
                'random_seed': random_seed,
            })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class dt_def(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "dt-robustness_l2"
        attacks = ['RBA_Exact_DT']

        grid_params = []
        for ds in tree_datasets:
            v, k = [ds], ds_eps[auto_var.get_var_shown_name("dataset", ds)]
            models = [
                'decision_tree_d5',
                f'adv_decision_tree_d5_{k}',
                f'robust_decision_tree_d5_{k}',
                f'advPruning_decision_tree_d5_{k}',
            ]

            grid_params.append({
                'model': models,
                'ord': [ATTACK_NORM],
                'dataset': v,
                'attack': attacks,
                'random_seed': random_seed,
            })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class rf_def(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "rf-robustness_l2"
        attacks = ['RBA_Approx_RF_100']

        grid_params = []
        for ds in tree_datasets:
            v, k = [ds], ds_eps[auto_var.get_var_shown_name("dataset", ds)]
            models = [
                'random_forest_100_d5',
                f'adv_rf_100_{k}_d5',
                f'robust_rf_100_{k}_d5',
                f'advPruning_rf_100_{k}_d5',
            ]

            grid_params.append({
                'model': models,
                'ord': [ATTACK_NORM],
                'dataset': v,
                'attack': attacks,
                'random_seed': random_seed,
            })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class lr_def(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "dt-robustness_l2"
        attacks = ['pgd']

        grid_params = []
        for ds in tree_datasets:
            v, k = [ds], ds_eps[auto_var.get_var_shown_name("dataset", ds)]
            models = [
                'logistic_regression',
                f'adv_logistic_regression_{k}',
                f'advPruning_logistic_regression_{k}',
            ]

            grid_params.append({
                'model': models,
                'ord': [ATTACK_NORM],
                'dataset': v,
                'attack': attacks,
                'random_seed': random_seed,
            })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class mlp_def(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "dt-robustness_l2"
        attacks = ['pgd']

        grid_params = []
        for ds in tree_datasets:
            v, k = [ds], ds_eps[auto_var.get_var_shown_name("dataset", ds)]
            models = ['mlp', f'adv_mlp_{k}', f'advPruning_mlp_{k}']

            grid_params.append({
                'model': models,
                'ord': [ATTACK_NORM],
                'dataset': v,
                'attack': attacks,
                'random_seed': random_seed,
            })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)
