# hw_01_modeling


## Introduction
This project was created to organize a pipeline model for a multilabel classification dataset “Planet: Understanding the Amazon from Space”.

Every minute, the world loses an area of forest the size of 48 football fields. And deforestation in the Amazon Basin accounts for the largest share, contributing to reduced biodiversity, habitat loss, climate change, and other devastating effects. But better data about the location of deforestation and human encroachment on forests can help governments and local stakeholders respond more quickly and effectively.

Planet, designer and builder of the world’s largest constellation of Earth-imaging satellites, will soon be collecting daily imagery of the entire land surface of the earth at 3-5 meter resolution. 
While considerable research has been devoted to tracking changes in forests, it typically depends on course-resolution imagery from Landsat (30 meter pixels) or MODIS (250 meter pixels). This limits its effectiveness in areas where small-scale deforestation or forest degradation dominate.

Furthermore, these existing methods generally cannot differentiate between human causes of forest loss and natural causes. Higher resolution imagery has already been shown to be exceptionally good at this, but robust methods have not yet been developed for Planet imagery.

Our task is to create a multi-class classification model that can label satellite images with different classes of land cover and land use. The model will help to understand where, how and why deforestation occurs in the world and how to respond to it.

## Dataset description
The dataset contains satellite images of nature representing 17 classes.  Each photo is presented in jpg format. An example of each class can be found in the folder artifacts/class_examples.jpeg

## Project structure
First of all, you need to upload the project to your working directory using GIT. 
Run the following command: git clone https://gitlab.deepschool.ru/cvr-dec23/e.romanov/hw_01_modeling.git

The project structure contains the following modules:

- configs: config file foy your work.
- src: main modules used for modeling and inference.

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

3. Download dataset.
```bash
make download_dataset
```
4. Сustomize [config.yaml](configs/config.yaml) for you training model.

5. ClearML:
    - Register in [ClearML](https://app.community.clear.ml/).
    - [In your profile ClearML](https://app.community.clear.ml/profile) press "Create new credentials"
    - Run `clearml-init` in you console.
    - Fill out credentials

## Train model

```bash
make train
```

## Linter style code
You can also check your code using linters. Run the command:

```bash
make lint 
```
If you want to change the linter parameters, then configure [setup.cfg](setup.cfg) for yourself

