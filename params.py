
from utils import RobustExperiments

random_seed = list(range(1))

datasets = [
    #'iris',
    #'wine',
    #'digits_pca5',
    #'digits_pca25',
    #'covtype_3200',
    #'splice',
    #'abalone',
    #'mnist35_2200_pca5', 'fashion_mnist35_2200_pca5',
    #'ijcnn1_2200',
    #'mnist35_2200_pca25',
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
    #'splice',
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

robust_datasets = [ # binary
    #'abalone',
    'fourclass',
    'diabetes',
    'cancer',
    #'ijcnn1_2200',
    'mnist17_2200_pca25',
    #'mnist35_2200_pca25',
    'fashion_mnist06_2200_pca25',
    'fashion_mnist35_2200_pca25',
    'halfmoon_2200',
]

large_datasets = [
    #'fashion_mnist35_10200_pca25',
    'fashion_mnist06_10200_pca25',
    'mnist35_10200_pca25',
    'mnist17_10200_pca25',
]

small_datasets = [
    'fashion_mnist35_300_pca5', 'mnist35_300_pca5',
    'fashion_mnist06_300_pca5', 'mnist17_300_pca5',
    'halfmoon_300'
]

class compare_nns(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "compare_nns"
        grid_params = []
        for i in [1, 3, 5, 7]:
            grid_params.append({
                'model': ['knn%d' % i],
                'ord': ['inf'],
                'dataset': datasets,
                'attack': ['rev_nnopt_k%d_50_region' % i],
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
            'ord': ['inf'],
            'dataset': datasets,
            'attack': ['direct_k1', 'blackbox', 'nnopt_k1_all', ],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': ['knn3'],
            'ord': ['inf'],
            'dataset': datasets,
            'attack': ['direct_k3', 'blackbox', 'rev_nnopt_k3_50_region'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': ['decision_tree_d5'],
            'ord': ['inf'],
            'dataset': tree_datasets,
            'attack': ['dt_papernots', 'blackbox', 'dt_attack_opt'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': ['random_forest_100_d5'],
            'ord': ['inf'],
            'dataset': tree_datasets,
            'attack': ['blackbox', 'rf_attack_rev_100'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class parametric_defense(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "parametric_defense"
        grid_params = []
        #grid_params.append({
        #    'model': [
        #        'sklr',
        #        'adv_sklr_30',
        #        'advPruning_sklr_30',
        #    ],
        #    'ord': ['inf'],
        #    'dataset': tree_datasets,
        #    'attack': ['pgd'],
        #    'random_seed': random_seed,
        #})
        grid_params.append({
            'model': [
                'logistic_regression',
                'adv_logistic_regression_30',
                'advPruning_logistic_regression_30',
            ],
            'ord': ['inf'],
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
            'ord': ['inf'],
            'dataset': tree_datasets,
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class compare_defense(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "compare_defense"
        grid_params = []
        grid_params.append({
            'model': ['knn1', 'adv_nn_k1_30', 'robustv2_nn_k1_30', 'advPruning_nn_k1_30'],
            'ord': ['inf'],
            'dataset': datasets,
            'attack': ['nnopt_k1_all'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': ['knn3', 'adv_nn_k3_30', 'advPruning_nn_k3_30'],
            'ord': ['inf'],
            'dataset': datasets,
            'attack': ['rev_nnopt_k3_50_region'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'decision_tree_d5',
                'adv_decision_tree_d5_30',
                'robust_decision_tree_d5_30',
                'advPruning_decision_tree_d5_30',
            ],
            'ord': ['inf'],
            'dataset': tree_datasets,
            'attack': ['dt_attack_opt'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'random_forest_100_d5',
                'adv_rf_100_30_d5',
                'robust_rf_100_30_d5',
                'advPruning_rf_100_30_d5',
            ],
            'ord': ['inf'],
            'dataset': tree_datasets,
            'attack': ['rf_attack_rev_100'],
            'random_seed': random_seed,
        })
        #grid_params.append({
        #    'model': [
        #        'mlp',
        #        'adv_mlp_30',
        #        'advPruning_mlp_30',
        #    ],
        #    'ord': ['inf'],
        #    'dataset': tree_datasets,
        #    'attack': ['pgd'],
        #    'random_seed': random_seed,
        #})
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class tst_scores(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "tst_scores"
        grid_params = []
        grid_params.append({
            'model': [
                #'adv_nn_k1_10', 'adv_nn_k1_30', 'adv_nn_k1_50',
                'knn1', 'advPruning_nn_k1_10', 'advPruning_nn_k1_30', 'advPruning_nn_k1_50',
                #'robustv2_nn_k1_10', 'robustv2_nn_k1_30', 'robustv2_nn_k1_50',
            ],
            'ord': ['inf'],
            'dataset': datasets,
            'attack': ['nnopt_k1_all'], #, 'blackbox'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                #'adv_nn_k1_10', 'adv_nn_k1_30', 'adv_nn_k1_50',
                'knn3', 'advPruning_nn_k3_10', 'advPruning_nn_k3_30', 'advPruning_nn_k3_50',
                #'robustv2_nn_k1_10', 'robustv2_nn_k1_30', 'robustv2_nn_k1_50',
            ],
            'ord': ['inf'],
            'dataset': datasets,
            'attack': ['rev_nnopt_k3_50_region'], #, 'blackbox'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'decision_tree_d5',
                'advPruning_decision_tree_d5_10', 'advPruning_decision_tree_d5_30', 'advPruning_decision_tree_d5_50',
            ],
            'ord': ['inf'],
            'dataset': tree_datasets,
            'attack': ['dt_attack_opt'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'random_forest_100_d5',
                'advPruning_rf_100_10_d5', 'advPruning_rf_100_30_d5', 'advPruning_rf_100_50_d5',
            ],
            'ord': ['inf'],
            'dataset': tree_datasets,
            'attack': ['rf_attack_rev_100'],
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
            'ord': ['inf'],
            'dataset': datasets,
            'attack': ['nnopt_k1_all'],#, 'blackbox'],
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
            'ord': ['inf'],
            'dataset': datasets,
            'attack': ['rev_nnopt_k3_50_region'],#, 'blackbox'],
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
        attacks = ['rf_attack_rev_100']

        grid_params = []
        grid_params.append({
            'model': models,
            'ord': ['inf'],
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
        attacks = ['dt_attack_opt']

        grid_params = []
        grid_params.append({
            'model': models,
            'ord': ['inf'],
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
            'ord': ['inf'],
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
            'ord': ['inf'],
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
            'ord': ['inf'],
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
            'ord': ['inf'],
            'dataset': tree_datasets,
            'attack': attacks,
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class opt_of_nnopt(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "Optimality3NNOPT"
        grid_params = [{
            'model': ['knn3'],
            'ord': ['inf'],
            'dataset': small_datasets,
            'attack': ['nnopt_k3_all',
                'rev_nnopt_k3_20_region', 'rev_nnopt_k3_50_region',],
            'random_seed': random_seed,
        }]
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class opt_of_rf_attack(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "optimality_rf"
        grid_params = []
        grid_params.append({
            'model': ['random_forest_3'],
            'ord': ['inf'],
            'dataset': small_datasets,
            'attack': ['rf_attack_all',
                'rf_attack_rev', 'rf_attack_rev_50',
                'rf_attack_rev_20', 'blackbox'
            ],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class rf_optimality(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "rf_optimality"
        grid_params = []
        grid_params.append({
            'model': ['random_forest_3_d5'],
            'ord': ['inf'],
            'dataset': small_datasets,
            'attack': [
                'rf_attack_all',
                'rf_attack_rev',
                'rf_attack_rev_100',
                'rf_attack_rev_15',
                'rf_attack_rev_10',
                'rf_attack_rev_5',
                'blackbox'
            ],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'robust_rf_3_20_d5', 'robust_rf_3_30_d5',
                'advPruning_rf_3_20_d5', 'advPruning_rf_3_20_d5',
                #'robustv2_rf_3_30_d5', 'robustv2_rf_3_30_d5',
            ],
            'ord': ['inf'],
            'dataset': small_datasets,
            'attack': ['rf_attack_all', 'blackbox'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class nn_optimality(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "nn_optimality_reduction"
        grid_params = []
        grid_params.append({
            'model': ['knn3'],
            'ord': ['inf'],
            'dataset': small_datasets,
            'attack': [
                'nnopt_k3_all',
                'rev_nnopt_k3_20_region', 'rev_nnopt_k3_50_region', 'blackbox'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'model': [
                'advPruning_nn_k3_30', 'advPruning_nn_k3_20',
                #'robustv2_nn_k3_20', 'robustv2_nn_k3_30',
            ],
            'ord': ['inf'],
            'dataset': small_datasets,
            'attack': ['nnopt_k3_all', 'blackbox'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class nn_k1_optimality_figs(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "nn_k1_optimality_figs"
        grid_params = []
        grid_params.append({
            'model': ['knn1'],
            'ord': ['inf'],
            'dataset': robust_datasets,
            'attack': ['blackbox', 'nnopt_k1_all',
                'rev_nnopt_k1_3_region',
                'rev_nnopt_k1_5_region',
                'rev_nnopt_k1_8_region',
                'rev_nnopt_k1_10_region',
                'rev_nnopt_k1_15_region',
                'rev_nnopt_k1_20_region',
                'rev_nnopt_k1_50_region',],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class nn_k3_optimality_figs(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "nn_k3_optimality_figs"
        grid_params = []
        grid_params.append({
            'model': ['knn3'],
            'ord': ['inf'],
            'dataset': small_datasets,
            'attack': ['blackbox',
                'nnopt_k3_all',
                'rev_nnopt_k3_3_region',
                'rev_nnopt_k3_5_region',
                'rev_nnopt_k3_8_region',
                'rev_nnopt_k3_10_region',
                'rev_nnopt_k3_15_region',
                'rev_nnopt_k3_20_region',
                'rev_nnopt_k3_50_region',],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class rf_optimality_figs(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "rf_optimality_figs"
        grid_params = []
        grid_params.append({
            'model': ['random_forest_3_d5'],
            'ord': ['inf'],
            'dataset': small_datasets,
            'attack': [
                'rf_attack_all',
                'rf_attack_rev_3',
                'rf_attack_rev_5',
                'rf_attack_rev_8',
                'rf_attack_rev_10',
                'rf_attack_rev_15',
                'rf_attack_rev_100',
                'rf_attack_rev',
                'blackbox'],
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
            'model': dt_models, 'attack': ['dt_attack_opt'],
            'dataset': tree_datasets, 'ord': ['inf'], 'random_seed': random_seed,
        }
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class nn_k1_robustness_figs(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "nn_k1_robustness_figs"
        nn_k1_models = ['knn1', 'advPruning_nn_k1_30',]
        grid_params = {
            'model': nn_k1_models, 'attack': ['nnopt_k1_all'],
            'dataset': datasets, 'ord': ['inf'], 'random_seed': random_seed,
        }
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class nn_k3_robustness_figs(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "nn_k3_robustness_figs"
        nn_k3_models = ['knn3', 'advPruning_nn_k3_30',]
        grid_params = {
            'model': nn_k3_models, 'attack': ['rev_nnopt_k3_50_region'],
            'dataset': datasets, 'ord': ['inf'], 'random_seed': random_seed,
        }
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)

class rf_robustness_figs(RobustExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "rf_robustness_figs"
        rf_models = ['random_forest_100_d5', 'robust_rf_100_30_d5',
                'advPruning_rf_100_30_d5',]
        grid_params = {
            'model': rf_models, 'attack': ['rf_attack_rev_100'],
            'dataset': tree_datasets, 'ord': ['inf'], 'random_seed': random_seed,
        }
        cls.grid_params = grid_params
        return RobustExperiments.__new__(cls, *args, **kwargs)
