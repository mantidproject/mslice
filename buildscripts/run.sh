#!/bin/bash

set -e
python setup.py flake8
xvfb-run python setup.py nosetests
