# CADET Documentation

This document provides instructions for building CADET documentation locally and for all releases and branches.

## Prerequisites

Ensure you have [mamba](https://mamba.readthedocs.io/en/latest/installation.html) installed on your system.

## Building Documentation Locally

1. **Navigate to the `doc` folder**:

    ```sh
    cd <root>/doc/
    ```

2. **Create and activate the documentation environment**:

    ```sh
    mamba env create -f ./environment.yml
    mamba activate cadet-core-docs
    ```

3. **Build the documentation**:

    ```sh
    sphinx-build -b html . build
    ```

    The output is in the `build` directory and can be opened with any browser.

## Building Documentation for All Releases and Branches

To build the documentation for all releases and the master branch, run:

```sh
sphinx-multiversion ./ ./build/ -D 'exhale_args.containmentFolder=${sourcedir}/api'
```

On Windows, use the following command:

```powershell
sphinx-multiversion ./ ./build/ -D exhale_args.containmentFolder=${sourcedir}/api
```
