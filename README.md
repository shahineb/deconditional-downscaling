# Deconditional Downscaling with Gaussian Processes

## Getting started

_Fill with brief description and command to run for immediate try out of repo_

- Run Swiss Roll experiment with matched dataset

Pick one of the configuration files listed under
```
experiments/swiss_roll/config/
├── exact_cme_process.yaml
├── bagged_gp.yaml
├── variational_cme_process.yaml
├── vbagg.yaml
└── gp_regression.yaml
```
and run from root directory

```bash
$ python experiments/swiss_roll/run_experiment.py --cfg=path/to/config/file --o=path/output/directory
```

- Run Swiss Roll experiment with unmatched dataset

Simply add `--unmatched` option to the above as
```bash
$ python experiments/swiss_roll/run_experiment.py --cfg=path/to/config/file --o=path/output/directory --unmatched
```


- Run Downscaling experiment

Pick one of the configuration files listed under
```
experiments/downscaling/config/
├── variational_cme_process_indiv_noise.yaml
├── vbagg.yaml
└── krigging.yaml
```
and run from root directory

```bash
$ python experiments/downscaling/run_experiment.py --cfg=path/to/config/file --o=path/output/directory
```



## Installation

Code implemented in Python 3.8.0

#### Setting up environment

Create and activate environment
```bash
$ pyenv virtualenv 3.8.0 venv
$ pyenv activate venv
$ (venv)
```

Install dependencies
```bash
$ (venv) pip install -r requirements.txt
```
