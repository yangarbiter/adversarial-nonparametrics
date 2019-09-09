
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

large_datasets = [
    'fashion_mnist35_10200_pca25',
    'fashion_mnist06_10200_pca25',
    'mnist17_10200_pca25',
]

small_datasets = [
    'fashion_mnist35_300_pca5',
    'fashion_mnist06_300_pca5',
    'mnist17_300_pca5',
    'halfmoon_300'
]

class compare_nns(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "compare_nns"
        grid_params = []
        for i in [1, 3, 5, 7]:
            grid_params.append({
                'model': ['knn%d' % i],
                'ord': [ATTACK_NORM],
                'dataset': datasets,
                'attack': ['RBA_Approx_KNN_k%d_50' % i],
                'random_seed': random_seed,
            })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class compare_attacks(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "compare_attacks"
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
            #'model': ['random_forest_200_d10'],
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': ['blackbox', 'RBA_Approx_RF_100'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class parametric_defense(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "parametric_defense"
        grid_params = []
        grid_params.append({
            'model': [
                'logistic_regression',
                'adv_logistic_regression_30',
                'advPruning_logistic_regression_30',
            ],
            'ord': [ATTACK_NORM],
            'dataset': tree_datasets,
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'mlp',
                'adv_mlp_30',
                'advPruning_mlp_30',
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
        cls.name = "compare_defense"
        def_strength = 50
        grid_params = []
        grid_params.append({
            #'model': ['knn1', 'adv_nn_k1_30', 'robustv2_nn_k1_30', 'advPruning_nn_k1_30'],
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
        cls.name = "tst_scores"
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

class nn_k1_robustness(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "1nn_robustness"
        grid_params = []
        grid_params.append({
            'model': [
                'knn1',
                'advPruning_nn_k1_10', 'advPruning_nn_k1_30', 'advPruning_nn_k1_50',
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
        cls.name = "3nn_robustness"
        grid_params = []
        grid_params.append({
            'model': [
                'knn3',
                'advPruning_nn_k3_10', 'advPruning_nn_k3_30', 'advPruning_nn_k3_50',
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
        cls.name = "rf-robustness"
        models = [
            'random_forest_100_d5',
            #'adv_rf_100_10_d5', 'adv_rf_100_30_d5', 'adv_rf_100_50_d5',
            #'robust_rf_100_10_d5', 'robust_rf_100_30_d5', 'robust_rf_100_50_d5',
            'advPruning_rf_100_10_d5', 'advPruning_rf_100_30_d5', 'advPruning_rf_100_50_d5',
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
        cls.name = "dt-robustness"
        models = [
            'decision_tree_d5',
            #'robust_decision_tree_d5_10', 'robust_decision_tree_d5_30', 'robust_decision_tree_d5_50',
            'advPruning_decision_tree_d5_10', 'advPruning_decision_tree_d5_30', 'advPruning_decision_tree_d5_50',
            #'robustv2_decision_tree_d5_10', 'robustv2_decision_tree_d5_30', 'advPruning_decision_tree_d5_50',
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
        cls.name = "lr-ap-robustness"
        models = [
            'logistic_regression',
            'advPruning_logistic_regression_10',
            'advPruning_logistic_regression_30',
            'advPruning_logistic_regression_50',
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
        cls.name = "lr-at-robustness"
        models = [
            'logistic_regression',
            'adv_logistic_regression_10',
            'adv_logistic_regression_30',
            'adv_logistic_regression_50',
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
        cls.name = "mlp-ap-robustness"
        models = [
            'mlp',
            'advPruning_mlp_10', 'advPruning_mlp_30', 'advPruning_mlp_50',
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
        cls.name = "mlp-at-robustness"
        models = [
            'mlp',
            'adv_mlp_10', 'adv_mlp_30', 'adv_mlp_50',
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

class dt_robustness_figs(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        dt_models = [
            'decision_tree_d5',
            'robust_decision_tree_d5_30',
            'advPruning_decision_tree_d5_30',
        ]

        cls.name = "dt_robustness_figs"
        grid_params = {
            'model': dt_models, 'attack': ['RBA_Exact_DT'],
            'dataset': tree_datasets, 'ord': [ATTACK_NORM], 'random_seed': random_seed,
        }
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class nn_k1_robustness_figs(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "nn_k1_robustness_figs"
        nn_k1_models = ['knn1', 'advPruning_nn_k1_30',]
        grid_params = {
            'model': nn_k1_models, 'attack': ['RBA_Exact_KNN_k1'],
            'dataset': datasets, 'ord': [ATTACK_NORM], 'random_seed': random_seed,
        }
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class nn_k3_robustness_figs(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "nn_k3_robustness_figs"
        nn_k3_models = ['knn3', 'advPruning_nn_k3_30',]
        grid_params = {
            'model': nn_k3_models, 'attack': ['RBA_Approx_KNN_k3_50'],
            'dataset': datasets, 'ord': [ATTACK_NORM], 'random_seed': random_seed,
        }
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class rf_robustness_figs(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "rf_robustness_figs"
        rf_models = ['random_forest_100_d5', 'robust_rf_100_30_d5',
                'advPruning_rf_100_30_d5',]
        grid_params = {
            'model': rf_models, 'attack': ['RBA_Approx_RF_100'],
            'dataset': tree_datasets, 'ord': [ATTACK_NORM], 'random_seed': random_seed,
        }
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)
