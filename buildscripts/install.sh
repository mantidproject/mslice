#!/bin/bash

# Install mslice requirements
pip install -r setup-requirements.txt -r install-requirements.txt -r test-requirements.txt coveralls
# Install X virtual framebuffer to emulate X server
apt-get install -y xvfb
