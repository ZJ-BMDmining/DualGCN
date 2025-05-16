# DualGCN-GE: Integration of spatiotemporal representations from whole-blood expression data with dual-view graph convolution network to identify Parkinson's Disease subtypes

Based on the architecture of multi-view graph learning, this study proposes the DualGCN-GE method to detect various PD subtypes, by integrating spatial and temporal patterns from disease-associated transcriptomic data.

## Architecture

![Fig1_GE-DualGCN-Diagram](./figures/Fig1_GE-DualGCN-Diagram.jpg)

## Install

To use DualGCN-GE  you must make sure that your python version is greater than 3.6. 

```
$ pip install -r requirements.txt
```

pytorch-cuda 11.8; 

torch-cluster 1.6.1+pt20cu118;

torch-geometric 2.2.0; 

torch-optimizer  0.3.0; 

torch-scatter 2.1.1+pt20cu118; 

torch-sparse  0.6.17+pt20cu118; 

torch-spline-conv 1.2.2+pt20cu118; 

numpy 1.23.5; 

shap  0.37.0; 

scanpy 1.10.3; 

scikit-learn 1.2.1

## Data availability

The dataset can be accessed from the [AMP-PD](https://amp-pd.org/) website.

## Usage

```
python   DualGCN.py -in <dataset.npz> -out <outputfolder> -bs <batch_size>
```

# License

This project is licensed under the MIT License - see the LICENSE file for details
