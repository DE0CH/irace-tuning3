from enum import Enum
from typing import Iterable, Union
from .errors import irace_assert, check_numbers
from .expressions import Expr, True_
import re

def guess_switch(name: str):
    if not re.match("^[a-z_]+$", name):
        raise NotImplementedError(f"unable to guess the switch for {name}")
    return "--" + name.replace("_", "-")

class ParameterType(Enum):
    INTEGER = 'i'
    REAL = 'r'
    ORDINAL = 'o'
    CATEGORICAL = 'c'
    INTEGER_LOG = 'i,log'
    REAL_LOG = 'r,log'

class ParameterDomain:
    pass

class Integer(ParameterDomain):
    def __init__(self, start, end, log=False):
        self.start = start
        self.end = end
        self.type = ParameterType.INTEGER_LOG if log else ParameterType.INTEGER
        check_numbers(start, end, log)
        irace_assert((isinstance(start, int) or isinstance(start, Expr)) and (isinstance(end, int) or isinstance(end, Expr)), "bounds must be integers or expressions")
    
    def __repr__(self):
        return f"{self.type.value} ({self.start}, {self.end})"

class Real(ParameterDomain):
    def __init__(self, start, end, log=False, digit = 4):
        start = float(start) if isinstance(start, int) else start
        self.start = start
        end = float(end) if isinstance(end, int) else end
        self.end = end
        self.type = ParameterType.REAL_LOG if log else ParameterType.REAL
        self.digit = digit
        check_numbers(start, end, log)
        irace_assert((isinstance(start, float) or isinstance(start, Expr)) and (isinstance(end, float) or isinstance(end, Expr)), "bounds must be numbers or expressions")

    def __repr__(self):
        return f"{self.type.value} (%.{self.digit}f, %.{self.digit}f)" % (self.start, self.end)

class Categorical(ParameterDomain):
    def __init__(self, domain: Iterable = None):
        if domain:
            self.domain = list(domain)
            self.type = ParameterType.CATEGORICAL
            irace_assert(len(set(domain)) == len(list(domain)), "domain has duplicate elements")
            for d in domain:
                irace_assert(isinstance(d, Expr) or isinstance(d, str), "domain element must be either string or expression (irace.expressions.Expr)")
        else:
            self.domain = list()

    def add_element(self, element):
        self.domain.append(element)

    def __repr__(self):
        return f"{self.type.value} ({', '.join(map(repr, self.domain))})"

class Ordinal(ParameterDomain):
    def __init__(self, domain: Iterable = None):
        if domain:
            self.domain = list(domain)
            self.type = ParameterType.ORDINAL
            for d in domain:
                irace_assert(isinstance(d, Expr) or isinstance(d, str), "domain element must be either string or expression (irace.expressions.Expr)")
            irace_assert(len(set(self.domain)) == len(self.domain), "domain has duplicate elements")
        else:
            self.domain = list()

    def add_element(self, element):
        self.domain.append(element)
    
    def __repr__(self):
        return f"{self.type.value} ({', '.join(map(repr, self.domain))})"

class Parameters:
    def __init__(self, initial_parameters = dict()):
        self.parameters: dict = initial_parameters
    
    def add_parameter(self, parameter_name, domain: ParameterDomain, condition = True_(), switch = ""):
        if parameter_name in self.parameters:
            raise ValueError(f"{parameter_name} already exists in parameters")
        self.parameters[parameter_name] = [switch, domain, condition]

    def set_condition(self, parameter_name, condition: Expr):
        self.parameters[parameter_name][2] = condition

    def guess_switch(self):
        for key in self.parameters:
            self.parameters[key][0] = guess_switch(key) + " "
        
    def as_string(self):
        lines = []
        for key in self.parameters:
            lines.append(f'{key} "{self.parameters[key][0]}" {repr(self.parameters[key][1])} | {repr(self.parameters[key][2])}')
        return '\n'.join(lines)
