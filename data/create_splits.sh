#!/usr/bin/env bash

set -e

python3 data_mfcc2split.py debug256 debug 128
python3 data_mfcc2split.py debug256 debug 256
python3 data_mfcc2split.py debug512 debug 512
python3 data_mfcc2split.py debug768 debug 768

python3 data_mfcc2split.py full256 full 128
python3 data_mfcc2split.py full256 full 256
python3 data_mfcc2split.py full512 full 512
python3 data_mfcc2split.py full768 full 768
