# FedBCD: A Communication-Efficient Collaborative Learning Framework for Distributed Features

## Abstract
We introduce a novel federated learning framework allowing multiple parties having different sets of attributes about the same user to jointly build models without exposing their raw data or model parameters. Conventional federated learning approaches are inefficient for cross-silo problems because they require the exchange of messages for gradient updates at every iteration, and raise security concerns over sharing such messages during learning. We propose a Federated Stochastic Block Coordinate Descent (FedBCD) algorithm, allowing each party to conduct multiple local updates before each communication to effectively reduce communication overhead. Under a practical security model, we show that parties cannot infer others' exact raw data (“deep leakage”) from collections of messages exchanged in our framework, regardless of the number of communication to be performed. Further, we provide convergence guarantees and empirical evaluations on a variety of tasks and datasets, demonstrating significant improvement inefficiency.

## Dataset

We use four datasets for experiments. 

- MNIST
- NUSWIDE
- MIMIC-III

You can adopt any dataset to run the code, but you have to prepare the code by yourself. 

## Getting Started

For convenience and quick experimental feedback, we simulate vertical federated learning without using any federated learning/distributed learning coding framework (e.g., FedML and FATE). You can straightforwardly adapt the code in this repo to these frameworks. 

## Requirements

This work was started in 2019, and we adopted TensorFlow 1.13 back then. You may either install TF 1.13 or transform our code to TF 2.0 or Pytorch.  

## Run example

- `vfl_learner` is the starting point for performing vertical federated learning training (no difference from the conventional neural network training).

- `vfl.py` includes the code for simulating guest and host parties. It also includes the vertical federated learning training procedure involving a guest and multiple hosts.

- `run_vfl_aue_two_party_demo.py` is running the experiments for a two-party VFL scenario, where both the guest and host adopt a one-layer FC neural network model. 

- `run_vfl_cnn_two_party_demo.py` is running the experiments for a two-party VFL scenario, where both the guest and host adopt a simple CNN model. 

For now, you can change the hyperparameters in `run_vfl_aue_two_party_demo.py` or `run_vfl_cnn_two_party_demo.py`, and run either file directly to start the training. We will add support for the command line later.

## Citation 

Accepted for publication in IEEE Transactions on Signal Processing, 2022.
Please kindly cite our paper if you find this code useful for your research.

```
@article{yang2022fedbcd,
  author={Liu, Yang and Zhang, Xinwei and Kang, Yan and Li, Liping and Chen, Tianjian and Hong, Mingyi and Yang, Qiang},
  journal={IEEE Transactions on Signal Processing}, 
  title={FedBCD: A Communication-Efficient Collaborative Learning Framework for Distributed Features}, 
  year={2022},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TSP.2022.3198176}}
```
