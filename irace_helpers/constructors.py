from .parameters import guess_switch
from typing import Callable
import os
import subprocess

def make_scenario_args(scenario, table: dict = dict(), default_translator: Callable[[str], str] = guess_switch):
    args = []
    for key in scenario:
        switch = table.get(key, default_translator(key))
        if not isinstance(scenario[key], str):
            raise ValueError("The value has to be a string")
        args.append(switch)
        args.append(scenario[key])
    return args

def get_irace_executable_path():
    return os.path.join(subprocess.check_output(['Rscript', '-e', "cat(system.file(package=\'irace\', \'bin\', mustWork=TRUE))"]).decode('utf-8'), 'irace')
