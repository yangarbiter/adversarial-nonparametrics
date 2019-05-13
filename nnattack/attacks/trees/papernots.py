"""
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ..base import AttackModel

# DT attacks
class decisionTreeNode:
    def __init__(self, node_id=None, input_component=None, threshold=None,left=None,right=None,output=[], parent=None):
        self.node_id = node_id
        self.input_component = input_component
        self.threshold = threshold
        self.left = left
        self.right = right
        self.output = output
        self.parent = parent

def tree_parser(clf):
    t = clf.tree_
    n_nodes = t.node_count
    children_left = t.children_left
    children_right = t.children_right
    feature = t.feature
    threshold = t.threshold
    values = t.__getstate__()['values']

    t_dict = {}
    stack = [(0, None)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_id = stack.pop()
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            left = children_left[node_id]
            right = children_right[node_id]
            input_component = feature[node_id]
            thres = threshold[node_id]
            output = values[node_id]

            stack.append((left, node_id))
            stack.append((right, node_id))
            if parent_id:
                t_dict[str(node_id)] = decisionTreeNode(str(node_id), input_component,
                                                   thres, str(left), str(right), output, str(parent_id))
            else:
                t_dict[str(node_id)] = decisionTreeNode(str(node_id), input_component,
                                                   thres, str(left), str(right), output, None)
        else:
            input_component = feature[node_id]
            thres = threshold[node_id]
            output = values[node_id]
            t_dict[str(node_id)] = decisionTreeNode(str(node_id), input_component,
                                                   thres, None, None, output, str(parent_id))
    return t_dict

def prediction(decisionTree_instance, sample, argmax=True, node_index=False):
    node = decisionTree_instance['0']
    while node.left or node.right:
        if float(sample[int(node.input_component)]) <= float(node.threshold):
            node = decisionTree_instance[node.left]
        else:
            node = decisionTree_instance[node.right]

    if argmax:
        return np.argmax(node.output)
    else:
        if node_index == False:
            return node.output
        else:
            return str(node.node_id)

def find_adv(decisionTree_instance, sample):
    legitimate_classification_node = decisionTree_instance[prediction(decisionTree_instance, sample, argmax=False, node_index=True)]
    legitimate_class = prediction(decisionTree_instance, sample, argmax=True)
    ancestor = legitimate_classification_node
    adv_node = legitimate_classification_node
    previous_ancestor = ancestor
    while np.argmax(adv_node.output) == legitimate_class and ancestor.parent:
        # is adv node on the left of its parent?
        list_components_left = [] #list of nodes where we went left
        list_components_right = [] #list of nodes where we went right
        if ancestor.node_id == decisionTree_instance[ancestor.parent].left:
            list_components_right.append([decisionTree_instance[ancestor.parent].input_component,
                                          decisionTree_instance[ancestor.parent].threshold])
            adv_node = decisionTree_instance[decisionTree_instance[ancestor.parent].right]
        else: # no, it is on the right
            list_components_left.append([decisionTree_instance[ancestor.parent].input_component,
                                         decisionTree_instance[ancestor.parent].threshold])
            adv_node = decisionTree_instance[decisionTree_instance[ancestor.parent].left]
        if adv_node.input_component:
            list_components_left.append([adv_node.input_component, adv_node.threshold])
        while adv_node.left or adv_node.right:
            adv_node = decisionTree_instance[adv_node.left]
            if adv_node.input_component:
                list_components_left.append([adv_node.input_component, adv_node.threshold])
        previous_ancestor = ancestor
        ancestor = decisionTree_instance[ancestor.parent]
    return previous_ancestor, adv_node, list_components_left, list_components_right

class Papernots(AttackModel):
    # https://arxiv.org/pdf/1605.07277.pdf
    def __init__(self, clf: DecisionTreeClassifier, ord, random_state):
        super().__init__(ord=ord)
        self.clf = clf
        self.random_state = random_state

        self.tree_structure = tree_parser(clf)

    def fit(self, X, y):
        pass

    def perturb(self, X, y, eps=0.1):
        #if len(self.clf.tree_.feature) == 1 and self.clf.tree_.feature[0] == -2:
        #    # only root and root don't split
        #    rret = []
        #    if isinstance(eps, list):
        #        for ep in eps:
        #            rret.append(pert_X)
        #        return rret
        #    else:
        #        return pert_X

        pert_X = np.zeros_like(X)
        pred_y = self.clf.predict(X)

        for sample_id in range(len(X)):
            if pred_y[sample_id] != y[sample_id]:
               continue
            x = np.copy(X[sample_id])
            _, _, l, r = find_adv(self.tree_structure, x)
            for (pixel, thres) in l:
                if pixel > 0:
                    x[pixel] = min(x[pixel], thres - 1e-3)
            for (pixel, thres) in r:
                if pixel > 0:
                    x[pixel] = max(x[pixel], thres + 1e-3)
            pert_X[sample_id] = x - X[sample_id]

        self.perts = pert_X
        return self._pert_with_eps_constraint(pert_X, eps)
