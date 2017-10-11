#!/bin/bash

set -e
python setup.py flake8
python setup.py nosetests
