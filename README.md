# Conditional Mean Embedding Process

Code for [[to be inserted]] by Siu Chau, Shahine Bouabid and Dino Sejdinovic


# Getting started

_Fill with brief description and command to run for immediate try out of repo_

- Run Swiss Roll Experiment

Pick one of the configuration files listed under
```
experiments/swiss_roll/config/
├── exact_cme_process.yaml
├── linear_interpolation.yaml
├── variational_cme_process.yaml
└── vbagg.yaml
```

and run from root directory

```bash
$ python experiments/swiss_roll/run_experiment.py --cfg=path/to/config/file --o=path/output/directory
```


# Repository structure

_Fill with description of repository structure_


# Installation

Code implemented in Python 3.8.7

#### Setting up environment

Clone and go to repository
```bash
$ git clone https://github.com/Chau999/ContBagGP.git
$ cd ContBagGP
```

Create and activate environment
```bash
$ pyenv virtualenv 3.8.7 venv
$ pyenv activate venv
$ (venv)
```

Install dependencies
```bash
$ (venv) pip install -r requirements.txt
```
