/usr/bin/python /workspaces/irace-tuning3/target-irace/start.py --pyrfr_wrapper /workspaces/irace-tuning3/target_algorithms/surrogate/cplex_regions200/pyrfr_wrapper.cplex_regions200.par10.random.pkl --pyrfr_model /workspaces/irace-tuning3/target_algorithms/surrogate/cplex_regions200/pyrfr_model.cplex_regions200.par10.random.bin --config_space /workspaces/irace-tuning3/target_algorithms/surrogate/cplex_regions200/config_space.cplex_regions200.par10.random.pcs --inst_feat_dict /workspaces/irace-tuning3/target_algorithms/surrogate/cplex_regions200/inst_feat_dict.cplex_regions200.par10.random.json -- --target-runner /workspaces/irace-tuning3/target-irace/target_runner_cost.py --parameter-file /workspaces/irace-tuning3/target_instances/surrogate/cplex_regions200/config_space.cplex_regions200.par10.random.irace.txt --train-instances-file /workspaces/irace-tuning3/target_instances/surrogate/cplex_regions200/train-1.txt --train-instances-dir / --test-instances-file /workspaces/irace-tuning3/target_instances/surrogate/cplex_regions200/test-1.txt --test-instances-dir / --digits 10 --seed 819121060 --debug-level 2 --max-experiments 2000 --capping 1 --bound-max 5