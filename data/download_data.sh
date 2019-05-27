#!/usr/bin/env bash

if [[ ! -f maestro-v2.0.0.zip ]]; then
    echo "Dowloading maestro-v2.0.0.zip... Hold your horses, this might take a while!"
    wget https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/
    unzip maestro-v2.0.0.zip
else
    echo "maestro-v2.0.0.zip already exists"
fi
