#!/usr/bin/env bash

python3 data_mfcc2split.py debug256 debug 256
python3 data_mfcc2split.py debug512 debug 512
python3 data_mfcc2split.py debug768 debug 768
python3 data_mfcc2split.py debug1024 debug 1024
python3 data_mfcc2split.py debug1280 debug 1280
python3 data_mfcc2split.py debug1536 debug 1536

python3 data_mfcc2split.py full256 full 256
python3 data_mfcc2split.py full512 full 512
python3 data_mfcc2split.py full768 full 768
python3 data_mfcc2split.py full1024 full 1024
python3 data_mfcc2split.py full1280 full 1280
python3 data_mfcc2split.py full1536 full 1536