from main import eps_accuracy

random_seed = list(range(2))

datasets = [
    #'iris', 'wine',
    #'digits_pca5',
    'fourclass',
    'diabetes',
    'digits_pca25',
    #'abalone',
    'covtype_3200',
    #'mnist35_2200_pca5', 'fashion_mnist35_2200_pca5',
    'ijcnn1_3200',
    'mnist17_2200_pca25',
    'fashion_mnist35_2200_pca25',
    'fashion_mnist06_2200_pca25',
    'mnist35_2200_pca25',
    'halfmoon_2200',
]

robust_datasets = [ # binary
    #'abalone',
    'fourclass',
    'diabetes',
    'ijcnn1_3200',
    'mnist17_2200_pca25',
    'mnist35_2200_pca25',
    'fashion_mnist06_2200_pca25',
    'fashion_mnist35_2200_pca25',
    'halfmoon_2200',
]

large_datasets = [
    'fashion_mnist35_10200_pca25',
    'fashion_mnist06_10200_pca25',
    'mnist35_10200_pca25',
    'mnist17_10200_pca25',
]

small_datasets = [
    'fashion_mnist35_300_pca5', 'mnist35_300_pca5',
    'fashion_mnist06_300_pca5', 'mnist17_300_pca5',
    'halfmoon_300'
]

def compare_nns():
    exp_fn = eps_accuracy
    exp_name = "compare_nns"
    grid_param = []
    for i in [1, 3, 5, 7]:
        grid_param.append({
            'model': ['knn%d' % i],
            'ord': ['inf'],
            'dataset': datasets,
            'attack': ['rev_nnopt_k%d_50_region' % i],
            'random_seed': random_seed,
        })
    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param


def nn_k1_robustness():
    exp_fn = eps_accuracy
    exp_name = "1nn_robustness"
    grid_param = []
    grid_param.append({
        'model': ['knn1'],
        'ord': ['inf'],
        'dataset': datasets,
        'attack': ['nnopt_k1_all', 'blackbox'],
        'random_seed': random_seed,
    })
    grid_param.append({
        'model': [
            'robustv1_nn_k1_10', 'robustv1_nn_k1_30', 'robustv1_nn_k1_50',
            'robustv2_nn_k1_10', 'robustv2_nn_k1_30', 'robustv2_nn_k1_50',
        ],
        'ord': ['inf'],
        'dataset': robust_datasets,
        'attack': ['nnopt_k1_all', 'blackbox'],
        'random_seed': random_seed,
    })
    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def nn_k3_robustness():
    exp_fn = eps_accuracy
    exp_name = "3nn_robustness"
    grid_param = []
    grid_param.append({
        'model': ['knn3'],
        'ord': ['inf'],
        'dataset': datasets,
        'attack': ['rev_nnopt_k3_50_region', 'blackbox'],
        'random_seed': random_seed,
    })
    grid_param.append({
        'model': [
            'robustv1_nn_k3_10', 'robustv1_nn_k3_30', 'robustv1_nn_k3_50',
            'robustv2_nn_k3_10', 'robustv2_nn_k3_30', 'robustv1_nn_k3_50',
        ],
        'ord': ['inf'],
        'dataset': robust_datasets,
        'attack': ['rev_nnopt_k3_50_region', 'blackbox'],
        'random_seed': random_seed,
    })
    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def rf_robustness():
    exp_fn = eps_accuracy
    exp_name = "rf-robustness"

    models = [
        'random_forest_100_d5',
        'robust_rf_100_10_d5', 'robust_rf_100_30_d5', 'robust_rf_100_50_d5',
        'robustv1_rf_100_10_d5', 'robustv1_rf_100_30_d5', 'robustv1_rf_100_50_d5',
        'robustv2_rf_100_10_d5', 'robustv2_rf_100_30_d5', 'robustv2_rf_100_50_d5',
    ]

    grid_param = []
    grid_param.append({
        'model': models,
        'ord': ['inf'],
        'dataset': large_datasets,
        'attack': [
            'rf_attack_rev_100', 'blackbox',
        ],
        'random_seed': random_seed,
    })
    grid_param.append({
        'model': ['random_forest_100_d5'],
        'ord': ['inf'],
        'dataset': datasets,
        'attack': [
            'rf_attack_rev_100', 'blackbox',
        ],
        'random_seed': random_seed,
    })
    grid_param.append({
        'model': models,
        'ord': ['inf'],
        'dataset': robust_datasets,
        'attack': [
            'rf_attack_rev_100', 'blackbox',
        ],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def dt_robustness():
    exp_fn = eps_accuracy
    exp_name = "dt-robustness"
    models = [
        'decision_tree',
        'robust_decision_tree_10', 'robust_decision_tree_30', 'robust_decision_tree_50',
        'robustv1_decision_tree_10', 'robustv1_decision_tree_30', 'robustv1_decision_tree_50',
        'robustv2_decision_tree_10', 'robustv2_decision_tree_30', 'robustv1_decision_tree_50',
    ]
    models = [
        'decision_tree_d5',
        'robust_decision_tree_d5_10', 'robust_decision_tree_d5_30', 'robust_decision_tree_d5_50',
        'robustv1_decision_tree_d5_10', 'robustv1_decision_tree_d5_30', 'robustv1_decision_tree_d5_50',
        'robustv2_decision_tree_d5_10', 'robustv2_decision_tree_d5_30', 'robustv1_decision_tree_d5_50',
    ]

    grid_param = []
    grid_param.append({
        'model': models,
        'ord': ['inf'],
        'dataset': robust_datasets,
        'attack': [
            'dt_attack_opt', 'blackbox',
        ],
        'random_seed': random_seed,
    })
    grid_param.append({
        'model': models,
        'ord': ['inf'],
        'dataset': robust_datasets,
        'attack': [
            'dt_attack_opt', 'blackbox',
        ],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def opt_of_nnopt():
    exp_fn = eps_accuracy
    exp_name = "Optimality3NNOPT"
    grid_param = [{
        'model': ['knn3'],
        'ord': ['inf'],
        'dataset': small_datasets,
        'attack': ['nnopt_k3_all',
            'rev_nnopt_k3_20_region', 'rev_nnopt_k3_50_region',],
        'random_seed': random_seed,
    }]

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def opt_of_rf_attack():
    exp_fn = eps_accuracy
    exp_name = "optimality_rf"
    grid_param = []
    grid_param.append({
        'model': ['random_forest_3'],
        'ord': ['inf'],
        'dataset': small_datasets,
        'attack': ['rf_attack_all',
            'rf_attack_rev', 'rf_attack_rev_50',
            'rf_attack_rev_20', 'blackbox'
        ],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param


def rf_optimality():
    exp_fn = eps_accuracy
    exp_name = "optimality_reduction"
    grid_param = []
    grid_param.append({
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
    grid_param.append({
        'model': [
            'robust_rf_3_20_d5', 'robust_rf_3_30_d5',
            'robustv1_rf_3_20_d5', 'robustv2_rf_3_30_d5',
            'robustv1_rf_3_20_d5', 'robustv2_rf_3_30_d5',
        ],
        'ord': ['inf'],
        'dataset': small_datasets,
        'attack': ['rf_attack_all', 'blackbox'],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def nn_optimality():
    exp_fn = eps_accuracy
    exp_name = "nn_optimality_reduction"
    grid_param = []
    grid_param.append({
        'model': ['knn3'],
        'ord': ['inf'],
        'dataset': small_datasets,
        'attack': [
            'nnopt_k3_all',
            'rev_nnopt_k3_20_region', 'rev_nnopt_k3_50_region', 'blackbox'],
        'random_seed': random_seed,
    })
    grid_param.append({
        'model': [
            'robustv1_nn_k3_30', 'robustv2_nn_k3_30',
            'robustv1_nn_k3_20', 'robustv2_nn_k3_20',
        ],
        'ord': ['inf'],
        'dataset': small_datasets,
        'attack': ['nnopt_k3_all', 'blackbox'],
        'random_seed': random_seed,
    })
    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def nn_k1_optimality_figs():
    exp_fn = eps_accuracy
    exp_name = "nn_k1_optimality_figs"
    grid_param = []
    grid_param.append({
        'model': ['knn1'],
        'ord': ['inf'],
        'dataset': robust_datasets,
        'attack': ['blackbox', 'nnopt_k1_all', 'rev_nnopt_k1_3_region',
            'rev_nnopt_k1_5_region', 'rev_nnopt_k1_10_region',
            'rev_nnopt_k1_15_region', 'rev_nnopt_k1_20_region',
            'rev_nnopt_k1_50_region',],
        'random_seed': random_seed,
    })
    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def nn_k3_optimality_figs():
    exp_fn = eps_accuracy
    exp_name = "nn_k3_optimality_figs"
    grid_param = []
    grid_param.append({
        'model': ['knn3'],
        'ord': ['inf'],
        'dataset': small_datasets,
        'attack': ['blackbox', 'nnopt_k3_all', 'rev_nnopt_k3_3_region',
            'rev_nnopt_k3_5_region', 'rev_nnopt_k3_10_region',
            'rev_nnopt_k3_15_region', 'rev_nnopt_k3_20_region',
            'rev_nnopt_k3_50_region',],
        'random_seed': random_seed,
    })
    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def rf_optimality_figs():
    exp_fn = eps_accuracy
    exp_name = "rf_optimality_figs"
    grid_param = []
    grid_param.append({
        'model': ['random_forest_3_d5'],
        'ord': ['inf'],
        'dataset': small_datasets,
        'attack': ['rf_attack_all', 'rf_attack_rev', 'rf_attack_rev_100',
            'rf_attack_rev_15', 'rf_attack_rev_10', 'rf_attack_rev_5',
            'blackbox'],
        'random_seed': random_seed,
    })
    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def dt_robustness_figs():
    dt_models = [
        'decision_tree_d5',
        'robust_decision_tree_d5_30',
        'robustv1_decision_tree_d5_30',
    ]

    exp_fn = eps_accuracy
    exp_name = "dt_robustness_figs"
    grid_param = {
        'model': dt_models, 'attack': ['dt_attack_opt'],
        'dataset': robust_datasets, 'ord': ['inf'], 'random_seed': random_seed,
    }
    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def nn_k1_robustness_figs():
    exp_fn = eps_accuracy
    exp_name = "nn_k1_robustness_figs"
    nn_k1_models = ['knn1', 'robustv1_nn_k1_30',]
    grid_param = {
        'model': nn_k1_models, 'attack': ['nnopt_k1_all'],
        'dataset': robust_datasets, 'ord': ['inf'], 'random_seed': random_seed,
    }
    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def nn_k3_robustness_figs():
    exp_fn = eps_accuracy
    exp_name = "nn_k3_robustness_figs"
    nn_k3_models = ['knn3', 'robustv1_nn_k3_30',]
    grid_param = {
        'model': nn_k3_models, 'attack': ['rev_nnopt_k3_50_region'],
        'dataset': robust_datasets, 'ord': ['inf'], 'random_seed': random_seed,
    }
    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param
