#!/bin/bash

# Install Python core module dependencies
pip install --no-cache-dir -r transcribe-app/src/core/requirements.txt

# Execute python core module
python3 -m transcribe-app.src.core