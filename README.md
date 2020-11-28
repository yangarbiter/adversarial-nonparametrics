# Robustness for Non-Parametric Classification: A Generic Attack and Defense

This repo contains the implementation of experiments in the paper

[Robustness for Non-Parametric Classification: A Generic Attack and Defense](https://arxiv.org/abs/1906.03310)

Authors: [Yao-Yuan Yang](http://yyyang.me/)\*, [Cyrus Rashtchian](http://www.cyrusrashtchian.com/)\*, [Yizhen Wang](http://cseweb.ucsd.edu/~yiw248/), [Kamalika Chaudhuri](http://cseweb.ucsd.edu/~kamalika/) (* equal contribution)

Appeared in AISTATS 2020 ([link to the presentation](https://aistats2020.net/poster_334.html))

#### Abstract

Adversarial examples have received a great deal of recent attention because of their potential to uncover security flaws in machine learning systems. However, most prior work on adversarial examples has been on parametric classifiers, for which generic attack and defense methods are known; non-parametric methods have been only considered on an ad-hoc or classifier-specific basis. In this work, we take a holistic look at adversarial examples for non-parametric methods. We first provide a general region-based attack that applies to a wide range of classifiers, including nearest neighbors, decision trees, and random forests. Motivated by the close connection between non-parametric methods and the Bayes Optimal classifier, we next exhibit a robust analogue to the Bayes Optimal, and we use it to motivate a novel and generic defense that we call adversarial pruning. We empirically show that the region-based attack and adversarial pruning defense are either better than or competitive with existing attacks and defenses for non-parametric methods, while being considerably more generally applicable.


## Installation

Python 3.6+

### Dependencies

```
pip install --upgrade -r requirements.txt
```

#### LP, QP Solvers

- Install gurobi: https://www.cvxpy.org/install/index.html#install-with-gurobi-support
- Install GLPK: https://www.cvxpy.org/install/index.html#install-with-cvxopt-and-glpk-support

### Install C-extensions
```
./setup.py build_ext -i
```

### for robust splitting
If you want to run robust splitting defense (https://arxiv.org/abs/1902.10660),
you'll have to install the modified scikit-learn in the package with the
following commend. For more installation detail, please reference to
https://github.com/scikit-learn/scikit-learn.

```
pip install --upgrade git+https://github.com/yangarbiter/scikit-learn.git@robustDT
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
