#!/usr/bin/env bash
set -e

# installing packages
sudo apt-get install -y python3 python3-pip python3-tk tmux python-pip python3-venv zip unzip wget
pip3 install --user virtualenv

# creating python virtualenv
virtualenv venv -p python3
source venv/bin/activate
pip3 --no-cache-dir install -r requirements.txt
