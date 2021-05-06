## Setup & installation

Install dependencies by installing and activating the environment:

```bash
conda env create -f environment.yml
conda activate ode
```

If you want to contribute, please install `pre-commit` to stay PEP8 compliant:

```bash
pre-commit install
```

To access the data you have to set up the git submodule for the
[COVID-19](https://github.com/CSSEGISandData/COVID-19) repository:

```bash
cd ode_nn/data/COVID-19
git submodule init
git submodule update
```



## TODO ITEMS:

- write a paper outline, envision the figures we want to produce/experiments we want to run
- write base classes and an API
  - Model(torch.nn.Module) - has a forward(SEUR_{t,i}) that spits out SEUR..._{t+1,i} (this class doesn't know about data)
  - higher level class that takes in a Model, spits out a loss (knows about data)
  - torch.utils.data.dataset class that takes in the CSVs from that other covid repo and delivers it in a tensor (and is cleaned)
    - cleaning includes: correct adjacently matrix, produce a mask from missing values
  - (w/ simulated data) write plotting routines (e.g. SEUR... vs time, gif of infected % spreading over US states heatmap, visualizing the MMD)
- a test suite (lol)
- a literature review especially w.r.t. parameters in compartmental models

Outstanding research questions we need to answer (this counts as work todo!):

- what exactly is the research question we are asking (e.g. making the best predictive model vs showcasing how to include demographic information)... leaning toward the latter
- will readers care about the MLE of the comparment model's parameters?
- how to we compare to previous models (what are the metrics of comparison)
- 

## Paper: 
Rui Wang, Danielle Maddix, Christos Faloutsos, Yuyang Wang, Rose Yu [Bridging Physics-based and Data-driven modeling for
Learning Dynamical Systems](https://arxiv.org/pdf/2011.10616.pdf), Annual Conference on Learning for Dynamics and Control (L4DC), 2021

## Abstract:
How can we learn a dynamical system to make forecasts, when some variables are unobserved? For instance, in COVID-19, we want to forecast the number of infected and death cases but we do not know the count of susceptible and exposed people. While mechanics compartment models are widely-used in epidemic modeling, data-driven models are emerging for disease forecasting. As a case study, we compare these two types of models for COVID-19 forecasting and notice that physics-based models significantly outperform deep learning models. We present a hybrid approach, AutoODE-COVID, which combines a novel compartmental model with automatic differentiation. Our method obtains a 57.4% reduction in mean absolute errors for 7-day ahead COVID-19 forecasting compared with the best deep learning competitor. To understand the inferior performance of deep learning, we investigate the generalization problem in forecasting. Through systematic experiments, we found that deep learning models fail to forecast under shifted distributions either in the data domain or the parameter domain. This calls attention to rethink generalization especially for learning dynamical systems.

## Description
1. ode_nn/: 
* DNN.py: Pytorch implementation of Seq2Seq, Auto-FC, Transformer, Neural ODE.
* Graph.py: Pytorch implementation of Graph Attention, Graph Convolution.
* AutoODE.py: Pytorch implementation of AutoODE(-COVID).
* train.py: data loaders, train epoch, validation epoch, test epoch functions.

3. Run_DSL.ipynb: train deep sequence models and graph neural nets.
4. Run_AutoODE.ipynb: train AutoODE-COVID. 
5. Evaluation.ipynb: evaluation functions and prediction visualization


## Requirement
* python 3.6
* pytorch 10.1
* matplotlib
* scipy
* numpy
* pandas
* dgl


## Cite
```
@inproceedings{wang2020bridging,
title={Bridging Physics-based and Data-driven modeling for Learning Dynamical Systems},
author={Rui Wang and Danielle Maddix and Christos Faloutsos and Yuyang Wang and Rose Yu},
journal={arXiv preprint arXiv:2011.10616},
year={2020}
}
```
