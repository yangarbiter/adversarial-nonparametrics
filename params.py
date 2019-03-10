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

def kernel_sub_clf():
    exp_fn = eps_accuracy
    random_seed = list(range(5))
    grid_param = []
    grid_param.append({
        'model': ['kernel_sub_tf', 'adv_kernel_sub_tf_c1_1',
                  'robustv1_kernel_sub_tf_c1_10'],
        'ord': ['inf'],
        'transformer': ['identity'],
        'dataset': ['fashion_mnist06_2000_pca25', 'mnist17_2000_pca25',
                    'mnist35_2000_pca25', 'fashion_mnist35_2000_pca25'],
        'attack': ['kernel_sub_pgd'],
        'random_seed': random_seed
    })
    grid_param.append({
        'model': ['kernel_sub_tf', 'adv_kernel_sub_tf_c1_1',
                  'robustv1_kernel_sub_tf_c1_3'],
        'ord': ['inf'],
        'transformer': ['identity'],
        'dataset': ['halfmoon2000'],
        'attack': ['kernel_sub_pgd'],
        'random_seed': random_seed
    })

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, grid_param, run_param

def adv_nn():
    exp_fn = eps_accuracy
    random_seed = list(range(1))
    grid_param = []
    grid_param.append({
        'model': ['adv_knn1_1', 'knn1', 'robustv1_nn_k1_5'],
        'ord': ['inf'],
        'transformer': ['identity'],
        'dataset': ['mnist35_2000_pca5'],
        'attack': ['rev_nnopt_k1_20'],
        'random_seed': random_seed,
    })
    grid_param.append({
        'model': ['adv_knn1_1', 'knn1', 'robustv1_nn_k1_5'],
        'ord': ['inf'],
        'transformer': ['identity'],
        'dataset': ['fashion_mnist06_2000_pca5', 'mnist17_2000_pca5',
                    'fashion_mnist35_2000_pca5'],
        'attack': ['rev_nnopt_k1_20'],
        'random_seed': random_seed,
    })
    grid_param.append({
        'model': ['adv_knn1_1', 'knn1', 'robustv1_nn_k1_1'],
        'ord': ['inf'],
        'transformer': ['identity'],
        'dataset': ['halfmoon2000'],
        'attack': ['rev_nnopt_k1_20'],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 1,}
    return exp_fn, grid_param, run_param

def adv_nn_k3():
    exp_fn = eps_accuracy
    random_seed = list(range(1))
    grid_param = []
    grid_param.append({
        'model': ['adv_knn3_1', 'knn3', 'robustv1_nn_k3_10'],
        'ord': ['inf', '2'],
        'transformer': ['identity'],
        'dataset': ['fashion_mnist06_2000_pca5', 'mnist17_2000_pca5',
                    'mnist35_2000_pca5', 'fashion_mnist35_2000_pca5'],
        'attack': ['rev_nnopt_k3_20'],
        'random_seed': random_seed,
    })
    grid_param.append({
        'model': ['adv_knn3_1', 'knn3', 'robustv1_nn_k3_10'],
        'ord': ['inf', '2'],
        'transformer': ['identity'],
        'dataset': ['halfmoon2000'],
        'attack': ['rev_nnopt_k3_20'],
        'random_seed': random_seed,
    })

    run_param = {'verbose': 1, 'n_jobs': 1,}
    return exp_fn, grid_param, run_param

def adv_sklr():
    exp_fn = eps_accuracy
    random_seed = list(range(5))
    grid_param = []
    grid_param.append({
        'model': ['adv_sklr_5', 'sklr', 'robustv1_sklr_regression_5'],
        'ord': ['inf'],
        'transformer': ['identity'],
        'dataset': ['fashion_mnist06_2000_pca25',
            'mnist17_2000_pca25', 'mnist35_2000_pca25',
            'fashion_mnist35_2000_pca25'
        ],
        'attack': ['sklr_opt'],
        'random_seed': random_seed
    })
    #grid_param.append({
    #    'model': ['adv_logistic_regression_1', 'logistic_regression',
    #        'robustv1_logistic_regression_1'],
    #    'ord': ['inf'],
    #    'transformer': ['identity'],
    #    'dataset': ['halfmoon2000'],
    #    'attack': ['pgd'],
    #    'random_seed': random_seed
    #})

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, grid_param, run_param

def adv_sklinsvc():
    exp_fn = eps_accuracy
    random_seed = list(range(5))
    grid_param = []
    grid_param.append({
        'model': ['adv_sklinsvc_5', 'sklinsvc', 'robustv1_sklinsvc_regression_5'],
        'ord': ['inf'],
        'transformer': ['identity'],
        'dataset': ['fashion_mnist06_2000_pca25',
            'mnist17_2000_pca25', 'mnist35_2000_pca25',
            'fashion_mnist35_2000_pca25'
        ],
        'attack': ['sklinsvc_opt'],
        'random_seed': random_seed
    })
    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, grid_param, run_param

def adv_lr():
    exp_fn = eps_accuracy
    random_seed = list(range(4))
    grid_param = []
    grid_param.append({
        'model': ['adv_logistic_regression_5', 'logistic_regression',
            'robustv1_logistic_regression_5'],
        'ord': ['inf'],
        'transformer': ['identity'],
        'dataset': ['fashion_mnist06_2000_pca25',
            'mnist17_2000_pca25', 'mnist35_2000_pca25', 'fashion_mnist35_2000_pca25'
        ],
        'attack': ['pgd'],
        'random_seed': random_seed
    })
    #grid_param.append({
    #    'model': ['adv_logistic_regression_1', 'logistic_regression',
    #        'robustv1_logistic_regression_1'],
    #    'ord': ['inf'],
    #    'transformer': ['identity'],
    #    'dataset': ['halfmoon2000'],
    #    'attack': ['pgd'],
    #    'random_seed': random_seed
    #})

    run_param = {'verbose': 1, 'n_jobs': 4,}
    return exp_fn, grid_param, run_param