export PATH="$(Rscript -e "cat(paste0(system.file(package='irace', 'bin', mustWork=TRUE), ':'))" 2> /dev/null)${PATH}"
irace --target-runner ./target-runner.py --parameter-file parameters.txt --max-experiments 96 --seed 123 --bound-max 1