# Resisting Over-Smoothing in Graph Neural Networks via Dual-Dimensional Decoupling

**The official pytorch implementation of "Resisting Over-Smoothing in Graph Neural Networks via Dual-Dimensional Decoupling" (ACM MM 24).**

![](./framework.svg)

> [Resisting Over-Smoothing in Graph Neural Networks via Dual-Dimensional Decoupling](https://marswhu.github.io/publications/files/MM24_DDCD.pdf)
>
> Wei Shen, Mang Ye, Wenke Huang
>
> Wuhan University
>
> **Abstract** Graph Neural Networks (GNNs) are widely employed to derive meaningful node representations from graphs. Despite their success, deep GNNs frequently grapple with the oversmoothing issue, where node representations become highly indistinguishable due to repeated aggregations. In this work, we consider the oversmoothing issue from two aspects of the node embedding space: dimension and instance. Specifically, while existing methods primarily concentrate on instance-level node relations to mitigate oversmoothing, we propose to mitigate oversmoothing at dimension level. We reveal the heightened information redundancy between dimensions which diminishes information diversity and impairs node differentiation in GNNs. Motivated by this insight, we propose the Dimension-Level Decoupling (DLD) to reduce dimension redundancy, enhancing dimensional-level node differentiation. Besides, at the instance level, the neglect of class differences leads to vague classification boundaries. Hence, we introduce the Instance-Level Class-Difference Decoupling (ICDD) that repels inter-class nodes and attracts intra-class nodes, improving the instance-level node discrimination with clear classification boundaries. Additionally, we introduce a novel evaluation metric that considers the impact of class differences on node distances, facilitating precise oversmoothing measurement. Extensive experiments demonstrate the effectiveness of our method Dual-Dimensional Class-Difference Decoupling (DDCD) across diverse scenarios.

## Requirements

python == 3.9

torch == 1.12.1

torch-geometric == 2.4.0

## Example Usage

For instance, to train GCN model with DDCD on Cora dataset, run:
```bash
python main.py --cuda_num=0 --dataset='Cora' --type_model='GCN' --type_norm='None' --DDCD True --alpha=0.006 --temperature=0.05
```


Hyperparameter explanations:

**--cuda_num:** The ID of GPU to be used.

**--dataset:** The training dataset.

**--type_model:** The type of GNN model.

**--type_norm:** The type of compared normalization methods. We include ['None', 'batch', 'pair', 'group'] for none normalization,  batch normalization, pair normalization and differentiable group normalization, respectively. When training with DDCD, set as 'None'.

**--DDCD:** Set as 'True' to apply DDCD to GNNs. Set as 'False' when applying other normalization methods.

**--alpha:** The hyperparameter that balances the trade-off of dimension decoupling.

**--temperature:** The hyperparameter that controls the contrastive force.

## Citation

```
@inproceedings{shen2024resisting,
  title={Resisting Over-Smoothing in Graph Neural Networks via Dual-Dimensional Decoupling},
  author={Shen, Wei and Ye, Mang and Huang, Wenke},
  booktitle={ACM Multimedia},
  year={2024}
}
```
