#!/bin/bash
# run this in the root directory of the project
set -e

python \
    tools/data_comparison/build_compare_cmds.py \
    train-records/cplex_regions200/data_train \
    tools/data_comparison/train-records/cplex_regions200/data_train \
    tools/data_comparison \
    target-irace/target_runner/target/release/target_runner \
    Instances/cplex_regions200-1.toml

python \
    tools/data_comparison/build_compare_cmds.py \
    train-records/cplex_regions200/data_traintest \
    tools/data_comparison/train-records/cplex_regions200/data_traintest \
    tools/data_comparison \
    target-irace/target_runner/target/release/target_runner \
    Instances/cplex_regions200-1.toml

python \
    tools/data_comparison/build_compare_cmds.py \
    train-records/cplex_regions200/data_val \
    tools/data_comparison/train-records/cplex_regions200/data_val \
     tools/data_comparison \
    target-irace/target_runner/target/release/target_runner \
    Instances/cplex_regions200-1.toml

python \
    tools/data_comparison/build_compare_cmds.py \
    train-records/minisat_randomk3/data_val \
    tools/data_comparison/train-records/minisat_randomk3/data_train \
    tools/data_comparison \
    target-irace/target_runner/target/release/target_runner \
    Instances/minisat_randomk3-1.toml

python \
    tools/data_comparison/build_compare_cmds.py \
    train-records/minisat_randomk3/data_val \
    tools/data_comparison/train-records/minisat_randomk3/data_traintest \
    tools/data_comparison \
    target-irace/target_runner/target/release/target_runner \
    Instances/minisat_randomk3-1.toml

python \
    tools/data_comparison/build_compare_cmds.py \
    train-records/minisat_randomk3/data_val \
    tools/data_comparison/train-records/minisat_randomk3/data_val \
    tools/data_comparison \
    target-irace/target_runner/target/release/target_runner \
    Instances/minisat_randomk3-1.toml
