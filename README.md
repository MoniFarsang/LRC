# Liquid-Resistance Liquid-Capacitance (LRC) Networks

This repository contains code for the paper "Liquid-Resistance Liquid-Capacitance Networks" presented at the [NeuroAI Workshop at NeurIPS 2024](https://neuroai-workshop.github.io/previous_years/2024/accepted-papers.html). The paper is available on [ArXiv](https://arxiv.org/pdf/2403.08791).

## Classification
To run the classification examples:
```
cd classification
python run_imdb.py --model LRC_sym_elastance --size 64
```
Model choices are LRCs with two types of elastance: `LRC_sym_elastance` and `LRC_asym_elastance`, and `lstm`, `mgu`, `gru`.

For the person localization, first download the dataset by running the `download_dataset.sh` script.

## Neural ODE
To run the neural ODE examples:
```
cd neuralODE
python run_ode.py --model lrc --lrc_type symmetric --data spiral --niters 1000
```
The data choices are:
`periodic_sinusodial`, `spiral`, `duffing`, `periodic_predator_prey`, `limited_predator_prey`, `nonlinear_predator_prey`.

Use `--viz True` for visualizing the progress of each validation step. 

Sinusoid           |  Spiral | Duffing
:-------------------------:|:-------------------------:|:-------------------------:
![](imgs/traj_phase_periodic_sinusodial.gif)  |  ![](imgs/traj_phase_spiral.gif) |  ![](imgs/traj_phase_duffing.gif)

Periodic Lotka-Volterra           |  Limited Lotka-Volterra  | Non-linear Lotka-Volterra
:-------------------------:|:-------------------------:|:-------------------------:
![](imgs/traj_phase_periodic_lv.gif)  |  ![](imgs/traj_phase_limited_lv.gif) |  ![](imgs/traj_phase_nonlinear_lv.gif)

# Citation
If you use this work, please cite our paper as follows:
```bibtex
@misc{farsang2024liquidresistanceliquidcapacitance,
      title={Liquid Resistance Liquid Capacitance Networks}, 
      author={Mónika Farsang and Sophie A. Neubauer and Radu Grosu},
      year={2024},
      eprint={2403.08791},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2403.08791}, 
}
```
