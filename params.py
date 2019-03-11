from main import eps_accuracy

def dt_attack():
    exp_fn = eps_accuracy
    random_seed = list(range(5))
    grid_param = []
    grid_param.append({
        'model': ['decision_tree', 'adv_decision_tree_1',
                  'robustv1_decision_tree_10'],
        'ord': ['inf'],
        'transformer': ['identity'],
        'dataset': ['fashion_mnist35_2000_pca25', 'mnist17_2000_pca25',
                    'mnist35_2000_pca25', 'fashion_mnist06_2000_pca25'],
        'attack': ['dt_attack_opt'],
        'random_seed': random_seed
    })
    grid_param.append({
        'model': ['decision_tree', 'adv_decision_tree_1',
                  'robustv1_decision_tree_1'],
        'ord': ['inf'],
        'transformer': ['identity'],
        'dataset': ['halfmoon2000'],
        'attack': ['dt_attack_opt'],
        'random_seed': random_seed
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, grid_param, run_param

def nn_k1():
    exp_fn = eps_accuracy
    random_seed = list(range(1))
    grid_param = []
    grid_param.append({
        'model': ['knn1'],
        'ord': ['inf'],
        'dataset': ['iris', 'wine', 'digits_pca5', 'abalone',
            'mnist35_2000_pca5'],
        'attack': ['rev_nnopt_k1_20', 'direct_k1', 'kernelsub_c100_pgd'],
        'random_seed': random_seed,
    })
    grid_param.append({
        'model': ['knn1'],
        'ord': ['inf'],
        'dataset': ['halfmoon_2000'],
        'attack': ['rev_nnopt_k1_20', 'direct_k1', 'kernelsub_c10_pgd'],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, grid_param, run_param

def robust_nn_k1():
    exp_fn = eps_accuracy
    random_seed = list(range(1))
    grid_param = []
    grid_param.append({
        'model': ['knn1'],
        'ord': ['2'],
        'dataset': ['iris', 'wine', 'digits_pca5', 'abalone',
            'mnist35_2000_pca5'],
        'attack': ['rev_nnopt_k1_20', 'direct_k1', 'kernelsub_c100_pgd'],
        'random_seed': random_seed,
    })
    grid_param.append({
        'model': ['knn1'],
        'ord': ['2'],
        'dataset': ['halfmoon_2000'],
        'attack': ['rev_nnopt_k1_20', 'direct_k1', 'kernelsub_c10_pgd'],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, grid_param, run_param

