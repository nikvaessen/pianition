#!/usr/bin/env bash

if [[ ! -f maestro-v2.0.0.zip ]]; then
    echo "Dowloading maestro-v2.0.0.zip... Hold your horses, this might take a while!"
    wget https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/
else
    echo "maestro-v2.0.0.zip already exists"
fi

if [[ ! -f maestro-v2.0.0.json ]]; then
    wget https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.json
else
    echo "maestro-v2.0.0.json already exists"
fi

if [[ ! -f maestro-v2.0.0.csv ]]; then
    wget https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.csv
else
    echo "maestro-v2.0.0.csv already exists"
fi
