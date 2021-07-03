## CADET Documentation


To build the documentation locally, install sphinx and other dependencies by running

```
pip install -r requirements.txt
```

Then, run `sphinx-multiversion ./ ./build/`. 

The above command only build the version of the documentation which is present on the master branch. In order to build the changes, made in the documentation, on the local machine use the following command:

`sphinx-build -b html . build` 

It is assumed this command is run in the scope of `doc` folder and it will put the output in a directory called `build`.

This documentation is published under https://cadet.github.io
Any changes to the documentation will automatically be pushed to the github-pages repository (https://github.com/cadet/cadet.github.io) using github actions.
