# Nonparametric Adversarial Attack and Defense

## Install

### Dependencies
```
pip install --upgrade -r requirements.txt
```

#### LP, QP Solvers
Install gruobi: https://www.cvxpy.org/install/index.html#install-with-gurobi-support
Install GLPK: https://www.cvxpy.org/install/index.html#install-with-cvxopt-and-glpk-support

### Install C-extensions
```
./setup.py build_ext -i
```

### for robust splitting
```
pip install --upgrade ./scikit-learn
```

## Example

- Run 3-NN using RBA-Approx searching 50 regions on dataset mnist 1 versus 7.
  The dataset has a total of 300 examples, 100 for training, from the 200
  leftout examples, select 100 corrected predicted data for purturbation.
  The feature dimension of the dataset is reduced to 25 using PCA.
```
python ./main.py --dataset mnist17_300_pca25 --model knn3 \
                 --attack rev_nnopt_k3_50_region --random_seed 0 --ord inf
```

- Train random forest with adversarial pruned (AP) dataset (defense strength is 0.3).
  The forest has 100 trees and maximum depth of 5.
  The attack is RBA-Approx searching 100 regions.
```
python ./main.py --dataset mnist17_300_pca25 --model robustv2_rf_100_30_d5 \
                 --attack rf_attack_rev_100 --random_seed 0 --ord inf
```
