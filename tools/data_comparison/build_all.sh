#!/bin/bash
# run this in the root directory of the project

python \
    tools/data_comparison/build_compare_cmds.py \
    cplex_regions200/data_train \
    tools/data_comparison/cplex_regions200/data_train \
    tools/data_comparison \
    target-irace/target_runner_cost.py \
    Instances/cplex_regions200-1.toml
