# Nonparametric Adversarial Attack and Defense

## Installation

Python 3.6+

### Dependencies

```
pip install --upgrade -r requirements.txt
```

#### LP, QP Solvers

- Install gruobi: https://www.cvxpy.org/install/index.html#install-with-gurobi-support
- Install GLPK: https://www.cvxpy.org/install/index.html#install-with-cvxopt-and-glpk-support

### Install C-extensions
```
./setup.py build_ext -i
```

### for robust splitting
If you want to run robust splitting defense (https://arxiv.org/abs/1902.10660),
you'll have to install the modified scikit-learn in the package with the
following commend. More installation detail please reference to
https://github.com/scikit-learn/scikit-learn.

```
pip install --upgrade
git+https://github.com/nonparametric-adversarial/scikit-learn.git
```

## Implementations

- [RBA-Approx-KNN](nnattack/attacks/nns/nn_attack.py): class KNNRegionBasedAttackApprox
- [RBA-Exact-KNN](nnattack/attacks/nns/nn_attack.py): class KNNRegionBasedAttackExact
- [RBA-Approx-RF](nnattack/attacks/trees/rf_attack.py): class KNNRegionBasedAttackApprox
- [RBA-Exact-RF](nnattack/attacks/trees/rf_attack.py): class KNNRegionBasedAttackApprox
- [RBA-Exact-DT](nnattack/attacks/trees/dt_opt.py): class KNNRegionBasedAttackExact
- [Adversarial Pruning](nnattack/models/defense.py)
- [Adversarial Pruning Decision Tree](nnattack/models/adversarial_dt.py): class AdversarialDt
- [Adversarial Pruning Random Forest](nnattack/models/adversarial_dt.py): class AdversarialRf
- [Adversarial Pruning Knn](nnattack/models/adversarial_knn.py): class AdversarialKnn

### Usage
```
usage: main.py [-h] [--no-hooks] --ord ORD --dataset DATASET --model MODEL
               --attack ATTACK --random_seed RANDOM_SEED

optional arguments:
  -h, --help            show this help message and exit
  --no-hooks            run without the hooks
  --ord ORD             Defines which distance measure to use for attack.
                        Options:
                          inf: L infinity norm
                          2: L2 norm
                          1: L1 norm
  --dataset DATASET     Defines the dataset to use
                        Options:
                          halfmoon_(?P<n_samples>\d+): halfmoon dataset, n_samples gives the number of samples
                          splice(?P<n_dims>_pca\d+)?: None
                          svmguide3: None
                          diabetes: None
                          fourclass: None
                          australian: None
                          cancer: None
                          covtypebin_(?P<n_samples>\d+): None
                          abalone: None
                          mnist17_(?P<n_samples>\d+)(?P<n_dims>_pca\d+)?: None
                          fashion_mnist06_(?P<n_samples>\d+)(?P<n_dims>_pca\d+)?: None
                          fashion_mnist35_(?P<n_samples>\d+)(?P<n_dims>_pca\d+)?: None
  --model MODEL         Defines which classifier to use.
                            The defense is implemented in this option.
                        Options:
                          random_forest_(?P<n_trees>\d+)(?P<depth>_d\d+)?: Random Forest Classifier
                          decision_tree(?P<depth>_d\d+)?: Original Decision Tree Classifier
                          (?P<train>[a-zA-Z0-9]+_)?decision_tree(?P<depth>_d\d+)?_(?P<eps>\d+):  Decision Tree classifier
                                train:
                                  adv: adversarial training
                                  robust: robust splitting
                                  robustv1: adversarial pruning
                                  robustv2: Wang's defense for 1-NN
                                eps: defense strength
                          (?P<train>[a-zA-Z0-9]+_)?rf_(?P<n_trees>\d+)_(?P<eps>\d+)(?P<depth>_d\d+)?: None
                          (?P<train>[a-zA-Z0-9]+)_kernel_sub_tf_c(?P<c>\d+)_(?P<eps>\d+): None
                          (?P<train>[a-zA-Z0-9]+_)?nn_k(?P<n_neighbors>\d+)_(?P<eps>\d+):  Nearest Neighbor classifier
                                train:
                                  adv: adversarial training
                                  robustv1: adversarial pruning
                                  robustv2: Wang's defense for 1-NN
                                eps: defense strength
                          knn(?P<n_neighbors>\d+): Original Nearest Neighbor classifier
                          kernel_sub_tf: None
                          (?P<train>[a-zA-Z0-9]+_)?mlp(?P<eps>_\d+)?: None
  --attack ATTACK       Defines which attack method to use.
                        Options:
                          rev_nnopt_k(?P<n_neighbors>\d+)_(?P<n_search>\d+)_region: RBA-Approx for Nearest Neighbor
                          gradient_based: Gradient Based Extension
                          nnopt_k(?P<n_neighbors>\d+)_all: RBA-Exact for nearest neighbor
                          kernelsub_c(?P<c>\d+)_(?P<attack>[a-zA-Z0-9]+): Kernel substitution model
                          direct_k(?P<n_neighbors>\d+): Direct Attack for Nearest Neighbor
                          pgd: Projected gradient descent attack
                          blackbox: Cheng's black box attack (BBox)
                          dt_papernot: Papernot's attack on decision tree
                          dt_attack_opt: RBA-Exact for Decision Tree
                          rf_attack_all: RBA-Exact for Random Forest
                          rf_attack_rev(?P<n_search>_\d+)?: RBA-Approx for Random Forest
  --random_seed RANDOM_SEED
```

## Examples

- To reproduce number in the paper, please set random_seed to 0 and set ord to
  inf.

1. Run 3-NN using RBA-Approx searching 50 regions on dataset mnist 1 versus 7.
   The dataset has a total of 2200 examples, 100 for training, from the 200
   leftout examples, select 100 corrected predicted data for purturbation.
   The feature dimension of the dataset is reduced to 25 using PCA.
```
python ./main.py --dataset mnist17_2200_pca25 --model knn3 \
                 --attack RBA_Approx_KNN_k3_50 --random_seed 0 --ord inf
```

2. Train random forest with adversarial pruned (AP) dataset (separation parameter r=0.3).
   The forest has 100 trees and maximum depth of 5.
   The attack is RBA-Approx searching 100 regions.
```
python ./main.py --dataset mnist17_10200_pca25 --model advPruning_rf_100_30_d5 \
                 --attack RBA_Approx_RF_100 --random_seed 0 --ord inf
```

3. Train 1-NN with adversarial pruned (AP) dataset (separation parameter r=0.3).
  The attack is RBA-Exact.
```
python ./main.py --dataset australian --model advPruning_nn_k1_30 \
                 --attack RBA_Exact_KNN_k1 --random_seed 0 --ord inf
```

4. Train 1-NN with adversarial training (AT) dataset (attack strength r=0.3).
  The attack is RBA-Exact.
```
python ./main.py --dataset australian --model adv_nn_k1_30 \
                 --attack RBA_Exact_KNN_k1 --random_seed 0 --ord inf
```

5. Train undefended 1-NN. The attack is RBA-Exact.
```
python ./main.py --dataset australian --model knn1 \
                 --attack RBA_Exact_KNN_k1 --random_seed 0 --ord inf
```

The improvement ration for knn1 with RBA-Exact on australian dataset  is the
number returned from 3 over the number returned from 4.
