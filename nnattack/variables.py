import os
import logging
from functools import partial

import numpy as np
from autovar import AutoVar
from autovar.base import RegisteringChoiceType, VariableClass, register_var
from autovar.hooks import (
    submit_parameter, upload_result, check_result_file_exist,
    save_parameter_to_file, save_result_to_file,
)

from .datasets import DatasetVarClass
from .models import ModelVarClass
from .attacks import AttackVarClass

def get_file_name(auto_var, name_only=False):
    dataset = auto_var.get_variable_name('dataset')
    model_name = auto_var.get_variable_name('model')
    attack_name = auto_var.get_variable_name('attack')
    ord = auto_var.get_variable_name('ord')
    random_seed = auto_var.get_variable_name('random_seed')

    name = "%s-%s-%s-rs%d" % (
        dataset, model_name, attack_name, random_seed)
    if ord == '1':
        name += "-l1"
    elif ord == '2':
        name += "-l2"
    elif ord == 'inf':
        name += "-linf"
    if name_only is False:
        base_dir = auto_var.settings['result_file_dir']
        name = os.path.join(base_dir, name)
    name = name.replace("_", "-")
    return name

class OrdVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines which distance measure to use for attack."""
    var_name = "ord"

    @register_var()
    @staticmethod
    def inf(auto_var):
        """L infinity norm"""
        return np.inf

    @register_var(argument='2')
    @staticmethod
    def l2(auto_var):
        """L2 norm"""
        return 2

    @register_var(argument='1')
    @staticmethod
    def l1(auto_var):
        """L1 norm"""
        return 1

auto_var = AutoVar(logging_level=logging.CRITICAL)

auto_var.add_variable_class(OrdVarClass())
auto_var.add_variable_class(DatasetVarClass())
auto_var.add_variable_class(ModelVarClass())
auto_var.add_variable_class(AttackVarClass())
auto_var.add_variable('random_seed', int)
