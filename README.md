# FedBCD: A Communication-Efficient Collaborative Learning Framework for Distributed Features

## Abstract
We introduce a novel federated learning framework allowing multiple parties having different sets of attributes about the same user to jointly build models without exposing their raw data or model parameters. Conventional federated learning approaches are inefficient for cross-silo problems because they require the exchange of messages for gradient updates at every iteration, and raise security concerns over sharing such messages during learning. We propose a Federated Stochastic Block Coordinate Descent (FedBCD) algorithm, allowing each party to conduct multiple local updates before each communication to effectively reduce communication overhead. Under a practical security model, we show that parties cannot infer others' exact raw data (“deep leakage”) from collections of messages exchanged in our framework, regardless of the number of communication to be performed. Further, we provide convergence guarantees and empirical evaluations on a variety of tasks and datasets, demonstrating significant improvement inefficiency.

## Dataset

### TBD

## Getting Started

### TBD

## Requirements

### TBD

## Run example

### TBD


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
