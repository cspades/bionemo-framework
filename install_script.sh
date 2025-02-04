#!/bin/bash
for sub in ./sub-packages/bionemo-*; do
    package_name=$(basename $sub)
    echo $package_name
    uv pip uninstall $package_name
    uv pip install --no-deps --no-build-isolation -e $sub
done
