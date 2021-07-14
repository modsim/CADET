## CADET Documentation


To build the documentation locally, install sphinx and other dependencies by running

```
pip install -r requirements.txt

```

Then, in the `doc` folder run:

`sphinx-build -b html . build` 

The output is in the `build` directory and can be opened with any browser.

To build the documentation for all releases and the master branch, run:

`sphinx-multiversion ./ ./build/`. 

This documentation is published under https://cadet.github.io
Any changes to the documentation will automatically be pushed to the github-pages repository (https://github.com/cadet/cadet.github.io) using github actions.
