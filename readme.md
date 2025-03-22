# DualGCN-GE: Integration of spatial-temporal representations from transcriptomics data with Dual-view Graph Convolutional Network to Identify Parkinson’s Disease Subtypes

Based on the architecture of multi-view graph learning, this study proposes the DualGCN-GE method to detect various PD subtypes, by integrating spatial and temporal patterns from disease-associated transcriptomic data.

## Architecture

![Fig1_GE-DualGCN-Diagram](./figures/Fig1_GE-DualGCN-Diagram.jpg)

## Install

To use DualGCN-GE  you must make sure that your python version is greater than 3.6. The best graphics card is RTx3060.

The required packages can be installed using the following command:

```
$ pip install -r requirements.txt
```

## Data availability

The dataset can be accessed from the [AMP-PD](https://amp-pd.org/) website.

## Usage

```
python   DualGCN.py -in <dataset.npz> -out <outputfolder> -bs <batch_size>
```
