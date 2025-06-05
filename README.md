# Liquid-Resistance Liquid-Capacitance (LRC) Networks

This repository contains code for the paper "Liquid-Resistance Liquid-Capacitance Networks" presented at the [NeuroAI Workshop at NeurIPS 2024](https://neuroai-workshop.github.io/previous_years/2024/accepted-papers.html). The paper is available on ArXiv [here](https://arxiv.org/pdf/2403.08791).

## Classification
To run the classification examples:
```
cd classification
python run_imdb.py --model LRC_sym_elastance --size 64
```

## Neural ODE
To run the neural ODE examples:
```
cd neuralODE
python run_ode.py --model lrc --lrc_type symmetric --data spiral --niters 1000
```
Use `--viz True` for visualizing the progress of each validation step. 

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
