import os
from functools import partial

import numpy as np
from autovar import AutoVar
from autovar.base import RegisteringChoiceType, VariableClass, register_var
from autovar.hooks import (
    submit_parameter, upload_result, check_result_file_exist,
    save_parameter_to_file, save_result_to_file,
)

from .datasets import DatasetVarClass
from .metrics import TransformerVarClass
from .models import ModelVarClass
from .attacks import AttackVarClass

def get_file_name(auto_var, name_only=False):
    transformer_name = auto_var.get_variable_value('transformer')
    dataset = auto_var.get_variable_value('dataset')
    model_name = auto_var.get_variable_value('model')
    attack_name = auto_var.get_variable_value('attack')
    ord = auto_var.get_variable_value('ord')
    random_seed = auto_var.get_variable_value('random_seed')

    name = "%s-%s-%s-%s-rs%d" % (
        dataset, model_name, attack_name, transformer_name, random_seed)
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
    var_name = "ord"

    @register_var()
    @staticmethod
    def inf(auto_var, var_value, inter_var):
        return np.inf

    @register_var(argument='2')
    @staticmethod
    def l2(auto_var, var_value, inter_var):
        return 2

    @register_var(argument='1')
    @staticmethod
    def l1(auto_var, var_value, inter_var):
        return 1

auto_var = AutoVar(
    before_experiment_hooks=[
        partial(check_result_file_exist, get_name_fn=get_file_name),
        #partial(save_parameter_to_file, get_name_fn=partial(get_file_name, name_only=False))
    ],
    after_experiment_hooks=[
        partial(save_result_to_file, get_name_fn=get_file_name)
    ],
    settings={
        'server_url': '',
        'result_file_dir': './results/'
    }
)

auto_var.add_variable_class(OrdVarClass())
auto_var.add_variable_class(DatasetVarClass())
auto_var.add_variable_class(TransformerVarClass())
auto_var.add_variable_class(ModelVarClass())
auto_var.add_variable_class(AttackVarClass())
auto_var.add_variable('test_size', float)
auto_var.add_variable('random_seed', int)
auto_var.add_variable('esp', float)
