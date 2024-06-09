# Model Documentation

1. The CNN is split into Optical and Event versions. There is also a Unified Event version and a Reduced version for Paper Ablation
2. ConvNeXtNet contains Event and Optical versions for both ConvNeXt and AlexNet
3. The LSTM is only an Event version
4. The SNN only has the Event and Unified Event versions
5. The FT Method provides a method to create data tables for the KNN method. Then KNN can be run to evaluate and create confusion matrices

## Convolutional Neural Network

The Optical version is exclusively compatible with the existing Nenad Dataset. The Optical version is for converted Nenad Data. The Unified version contains both Nenad Data and Won Lab Data. The reduced version operates the same as the regular version, but uses a reduced dataset for the sake of showing an ablation study.

*Note: Unified Refers to the combined Won Lab and Nenad Lab Data. This is used throughout the project to refer to the combined data.* 

## ConvNeXtNet

There is a parameter in this file to switch between pretrained and non-pretrained versions. AlexNet and ConvNeXt are both used and are available for comparison sake. The Fine Tune Layers can be easily accessed and changed if need be.

## LSTM

The LSTM is bog-standard, and explained in depth in the paper.

## Spiking Neural Network

The SNN contains Event and Unified Event versions. The hyper parameters are further explored in the paper, but the $\beta$ is the main SNN specific hyper parameter.

## Fourier Based Approach

Data is created according to signal processing based approaches as explored in testing file [psdFinal.ipynb](psdFinal.ipynb).

**See:** [ftScatter.py](ftScatter.py) for Data Creation

*Note 1: An output file is created for this data*

K-Nearest Neighbors is then run to determine the correct cluster.

**See:** [knn.py](knn.py) for this process

*Note 2: For Testing and Visualization purposes the file [scatterCheck.py](scatterCheck.py)* is provided
