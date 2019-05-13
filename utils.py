import os
import json
import logging
from typing import List, Dict, Union, Callable, Any

import numpy as np
import pandas as pd

from nnattack.variables import auto_var, get_file_name
from autovar import AutoVar
from main import eps_accuracy

logging.basicConfig(level=0)
tex_base = "./tex_files"

class Experiments():
    name: str
    experiment_fn: Callable[[AutoVar], Any]
    grid_params: Union[List[Dict[str, str]], Dict[str, str]]
    run_param: Dict[str, Any]

    def __init__(self):
        pass

    def __call__(self):
        return self.experiment_fn, self.name, self.grid_params, self.run_param

class RobustExperiments(Experiments):
    def __new__(cls, *args, **kwargs):
        # if attribute is function it will pass self as one of its argument
        cls.experiment_fn = lambda _, b: eps_accuracy(b)
        cls.run_param = {'verbose': 1, 'n_jobs': 4,}
        return Experiments.__new__(cls, *args, **kwargs)

def get_result(auto_var):
    file_name = get_file_name(auto_var, name_only=True).replace("_", "-")
    file_path = f"./results/{file_name}.json"
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r") as f:
            ret = json.load(f)
    except Exception as e:
        print("problem with %s" % file_path)
        raise e
    return ret

def params_to_dataframe(grid_param, columns=None):
    params, loaded_results = auto_var.run_grid_params(
            get_result, grid_param, with_hook=False, verbose=0, n_jobs=1)
    if columns is None:
        results = [r['results'] if isinstance(r, dict) else r for r in loaded_results]
    else:
        results = loaded_results

    params, results = zip(*[(params[i], results[i]) for i in range(len(params)) if results[i]])
    params, results = list(params), list(results)
    accs = []
    for i, param in enumerate(params):
        if columns is None:
            for r in results[i]:
                #params[i][f'eps_{r["eps"]:.2f}_trn'] = r['trn_acc']
                params[i][f'eps_{r["eps"]:.2f}_tst'] = r['tst_acc']
        else:
            for column in columns:
                if column not in results[i] and column != 'aug_len':
                    params[i][column] = np.nan
                else:
                    if column == 'avg_pert':
                        if 'single_label' in results[i] and results[i]['single_label']:
                            params[i][column] = np.nan
                            if 'missed_count' in results[i]['avg_pert']:
                                params[i]['missed_count'] = results[i]['avg_pert']['missed_count']
                            else:
                                params[i]['missed_count'] = 0
                        else:
                            params[i][column] = results[i][column]['avg']
                            if 'missed_count' in results[i]['avg_pert']:
                                params[i]['missed_count'] = results[i]['avg_pert']['missed_count']
                            else:
                                params[i]['missed_count'] = 0
                    elif column == 'aug_len':
                        if column not in results[i]:
                            params[i][column] = results[i]['trnX_len']
                        else:
                            params[i][column] = results[i][column]
                    else:
                        params[i][column] = results[i][column]

    df = pd.DataFrame(params)
    return df

def set_plot(fig, ax, ord=np.inf):
    fig.autofmt_xdate()
    ax.legend()
    ax.set_ylim(0, 1)
    #ax.legend(bbox_to_anchor=(1.5, 0., 0.5, 0.5))
    ax.legend(prop={'size': 16}, loc='upper right', frameon=True)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_ylabel('Accuracy', fontsize=20)
    xlabel = 'Perturbation distance'
    if ord == np.inf:
        ax.set_xlabel(xlabel + ' (Linf)', fontsize=20)
    else:
        ax.set_xlabel(xlabel, fontsize=20)

def write_to_tex(s, file_name):
    with open(os.path.join(tex_base, file_name), 'w') as f:
        f.write(s)


def union_param_key(grid_param, key):
    if isinstance(grid_param, list):
        ret = []
        for g in grid_param:
            for v in g[key]:
                if v not in ret:
                    ret.append(v)
        return ret
        #return set.union(*[set(g[key]) for g in grid_param])
    else:
        return grid_param[key]


def table_wrapper(table_df, table_name, caption=''):
    t = """
\\begin{table}[h!]
\\tiny
\\centering
\\setlength{\\tabcolsep}{2.0pt}
"""
    t += table_df.to_latex(escape=False)
    t += """\\caption{%s}
\\label{table:%s}
\\end{table}
""" % (caption, table_name)
    return t
