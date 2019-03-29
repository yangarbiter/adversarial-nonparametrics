from main import eps_accuracy

random_seed = list(range(4))

def dt_attack():
    exp_fn = eps_accuracy
    random_seed = list(range(2))
    grid_param = []
    grid_param.append({
        'model': ['decision_tree', 'adv_decision_tree_1',
                  'robustv1_decision_tree_10'],
        'ord': ['inf'],
        'dataset': ['fashion_mnist35_2000_pca15', 'mnist35_2000_pca15',
            'fashion_mnist06_2000_pca15', 'halfmoon_2000'],
        'attack': ['dt_attack_opt'],
        'random_seed': random_seed
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, grid_param, run_param

def rf_attack():
    exp_fn = eps_accuracy
    exp_name = "random_forest"
    grid_param = []
    grid_param.append({
        'model': ['random_forest_100'],
        'ord': ['inf'],
        'dataset': [
            'fashion_mnist35_2000_pca5',
            'mnist35_2000_pca5',
            'fashion_mnist06_2000_pca5',
            'fashion_mnist35_2000_pca10',
            'mnist35_2000_pca10',
            'fashion_mnist06_2000_pca10',
            'halfmoon_2000'
        ],
        'attack': [
            'rf_attack_rev_100',
            'rf_attack_rev_20',
            'blackbox'],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def rf500_attack():
    exp_fn = eps_accuracy
    exp_name = "rf500"
    grid_param = []
    grid_param.append({
        'model': ['random_forest_500'],
        'ord': ['inf'],
        'dataset': [
            'fashion_mnist35_2000_pca5',
            'mnist35_2000_pca5',
            'fashion_mnist06_2000_pca5',
            'halfmoon_2000'
        ],
        'attack': [
            'rf_attack_rev_100',
            'rf_attack_rev_20',
            'blackbox'],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def opt_of_rf_attack():
    exp_fn = eps_accuracy
    exp_name = "optimality_rf"
    grid_param = []
    grid_param.append({
        'model': ['random_forest_3'],
        'ord': ['inf'],
        'dataset': [
            'fashion_mnist35_200_pca5',
            'mnist35_200_pca5',
            'fashion_mnist06_200_pca5',
            'halfmoon_200'
        ],
        'attack': ['rf_attack_all',
            'rf_attack_rev', 'rf_attack_rev_50', 'rf_attack_rev_20', 'blackbox'],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def nn_k1():
    exp_fn = eps_accuracy
    exp_name = "1nn"
    grid_param = []
    grid_param.append({
        'model': ['knn1'],
        'ord': ['inf'],
        'dataset': ['iris', 'wine', 'digits_pca5'],
        'attack': ['rev_nnopt_k1_20', 'direct_k1', 'kernelsub_c1000_pgd', 'blackbox'],
        'random_seed': random_seed,
    })
    grid_param.append({
        'model': ['knn1'],
        'ord': ['inf'],
        'dataset': ['abalone',
            'mnist35_2000_pca5', 'fashion_mnist06_2000_pca5', 'fashion_mnist35_2000_pca5',
            #'mnist35_2000_pca15', 'fashion_mnist06_2000_pca15',
            'fashion_mnist35_2000_pca15',
        ],
        'attack': ['rev_nnopt_k1_20', 'rev_nnopt_k1_50', 'direct_k1', 'kernelsub_c10000_pgd', 'blackbox'],
        'random_seed': random_seed,
    })
    grid_param.append({
        'model': ['knn1'],
        'ord': ['inf'],
        'dataset': ['halfmoon_2000'],
        'attack': ['rev_nnopt_k1_20', 'rev_nnopt_k1_50', 'direct_k1', 'kernelsub_c1000_pgd', 'blackbox'],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def robust_nn_k1():
    exp_fn = eps_accuracy
    exp_name = "Robust1nn"
    grid_param = []
    grid_param.append({
        'model': ['robust1nn'],
        'ord': ['inf'],
        'dataset': ['abalone',
            'mnist35_2000_pca5', 'fashion_mnist06_2000_pca5'
            'mnist35_2000_pca15', 'fashion_mnist06_2000_pca15'
        ],
        'attack': ['rev_nnopt_k1_20', 'direct_k1', 'kernelsub_c100_pgd', 'blackbox'],
        'random_seed': random_seed,
    })
    grid_param.append({
        'model': ['robust1nn'],
        'ord': ['inf'],
        'dataset': ['halfmoon_2000'],
        'attack': ['rev_nnopt_k1_20', 'direct_k1', 'kernelsub_c10_pgd', 'blackbox'],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

datasets = [
    'iris', 'wine', 'digits_pca5', 'abalone',
    'mnist35_2000_pca5', 'fashion_mnist35_2000_pca5',
    'fashion_mnist06_2000_pca5',
    #'mnist35_2000_pca15', 'fashion_mnist35_2000_pca15',
    #'fashion_mnist06_2000_pca15',
    'halfmoon_2000',
]

def nn_k7():
    exp_fn = eps_accuracy
    exp_name = "7nn"
    grid_param = []
    grid_param.append({
        'model': ['knn7'],
        'ord': ['inf'],
        'dataset': datasets,
        'attack': [
            #'hybrid_nnopt_k7_10_20',
            'rev_nnopt_k7_20', 'rev_nnopt_k7_50',
            'rev_nnopt_k7_20_region', 'rev_nnopt_k7_50_region',
            'direct_k7',
            'blackbox',
            #'nnopt_k7_20', 'nnopt_k7_50'
        ],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def nn_k5():
    exp_fn = eps_accuracy
    exp_name = "5nn"
    grid_param = []
    grid_param.append({
        'model': ['knn5'],
        'ord': ['inf'],
        'dataset': datasets,
        'attack': ['rev_nnopt_k5_20', 'rev_nnopt_k5_50',
            'rev_nnopt_k5_20_region', 'rev_nnopt_k5_50_region', 'direct_k5',
            'blackbox',
            #'nnopt_k5_20', 'nnopt_k5_50'
        ],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param


def nn_k3():
    exp_fn = eps_accuracy
    exp_name = "3nn"
    grid_param = []
    grid_param.append({
        'model': ['knn3'],
        'ord': ['inf'],
        'dataset': datasets,
        'attack': [
            #'rev_nnopt_k3_20',
            'rev_nnopt_k3_50', 'direct_k3', 'blackbox'
            'rev_nnopt_k3_50_region',
        ],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def robust_nn_k3():
    exp_fn = eps_accuracy
    exp_name = "Robust-3nn"
    grid_param = []
    grid_param.append({
        'model': ['knn3'],
        'ord': ['inf'],
        'dataset': datasets,
        'attack': [
            #'rev_nnopt_k3_20',
            'rev_nnopt_k3_50', 'direct_k3', 'blackbox'
            'rev_nnopt_k3_50_region',
        ],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param

def opt_of_nnopt():
    exp_fn = eps_accuracy
    exp_name = "OptimalityNNOPT"
    grid_param = [{
        'model': ['knn3'],
        'ord': ['inf'],
        'dataset': ['fashion_mnist35_200_pca5', 'mnist35_200_pca5',
            #'fashion_mnist35_200_pca15', 'mnist35_200_pca15',
            'halfmoon_200'],
        'attack': ['nnopt_k3_all', 'rev_nnopt_k3_50', 'rev_nnopt_k3_20',
            'rev_nnopt_k3_20_region', 'rev_nnopt_k3_50_region',],
        'random_seed': random_seed,
    }]

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, exp_name, grid_param, run_param
