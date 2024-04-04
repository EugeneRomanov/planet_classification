# hw1_mod service

This repository allows you to deploy a service with [model](https://github.com/EugeneRomanov/planet_classification/tree/main/hw_01_modeling-dev) results for multiclass forecasting for the "Planet" dataset

## Preparing the environment
Makefile allows you to simply get started with this project.

1. Install environment.
```bash
make venv
```

2. Install requirements.
```bash
make install_requirements
```

## Run service

```bash
make run_app
```

## Build the image

```bash
make build
```

## Linter style code
You can also check your code using linters. Run the command:

```bash
make lint 
```
If you want to change the linter parameters, then configure [setup.cfg](setup.cfg) for yourself


## Service
The launched service is available via this [link](http://localhost:2444/docs)
