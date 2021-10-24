# Deconditional Downscaling with Gaussian Processes


<p align="center">
<img src="https://github.com/shahineb/deconditional-downscaling/blob/main/docs/img/figure_1.png" alt="figure" width="300"/>
 </p>

## Getting started

- Run Swiss Roll experiment with matched dataset

Pick one of the configuration files listed under
```
experiments/swiss_roll/config/
├── exact_cmp.yaml
├── bagged_gp.yaml
├── variational_cmp.yaml
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
├── variational_cmp_indiv_noise.yaml
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



## Reference
```
@inproceedings{ChaBouSej2021,
  title = {{Deconditional Downscaling with Gaussian Processes}},
  author = {Chau, Siu Lun and Bouabid, Shahine and Sejdinovic, Dino},
  year = {2021},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}
}
```
