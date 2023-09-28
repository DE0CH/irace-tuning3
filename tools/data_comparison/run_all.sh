#!/bin/sh

set -e
cd tools/data_comparison/cplex_regions200/data_train
bash cmds.sh
cd ../../../..
cd tools/data_comparison/cplex_regions200/data_val
bash cmds.sh
cd ../../../..
cd tools/data_comparison/cplex_regions200/data_traintest
bash cmds.sh
cd ../../../..
