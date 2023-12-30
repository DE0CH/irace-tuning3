#!/bin/sh

cd target-irace/target_runner
cargo build --release
cd ../..

pip3 install -r requirements.txt
