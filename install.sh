#!/bin/sh

cd target-irace/target_runner
cargo build --release
cd ../..

pip install -r requirements.txt
