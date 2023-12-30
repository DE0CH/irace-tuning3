# Compare Prediction with Actual

We compare if the prediction from the model is the same as the recorded data provided.

# Dependencies
- `python3`
- rust with `cargo`

# Installation

```
python3 -m venv venv
source venv/bin/activate
./install.sh
```

# Usage

## Configuration Files

.toml instance files TODO
settings.toml files TODO

## Command line

Usage:
```
irace --target-runner-launcher python3 --target-runner path/to/target-runner.py <remaining irace args>
```

Example:
```
irace --target-runner ../irace-tuning3/target-runner.py --parameter-file parameters.txt --max-experiments 96 --seed 123
```

The paths specified in the config files are all relative to the current working directory.

The location of the scripts for internal use (those not controlled by the user, e.g. the target runner for target irace) are relative to the directory of the environment variable IRACE_TUNING_PATH or the folder of `target-runner.py` script if the variable is not set.

## Caveat

Do not set the `boundMax` function, it is not supported and will cause undefined behavior.
