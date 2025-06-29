#!/bin/bash

# Create venv:
python3 -m venv python_env

# Load env:
source python_env/bin/activate

# Install jupyter:
#pip3 install --quiet --upgrade jupyterlab jupyterlab-git

# Set LD_LIBRARY_PATH for WSL cuda support.
LD_LIBRARY_PATH=/usr/lib/wsl/lib/ \
        jupyter lab \
        --no-browser
