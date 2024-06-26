# scRNA-seq-Labeler

## Overview

The exponentially increasing amount of RNA sequencing data following the development of RNA sequencing technology provides novel knowledge about high-resolution profiling of cell transcriptomes. However, one of the largest challenges with RNA-seq data is properly labeling data for analysis. Labeling the cell clusters of single-cell RNA-seq data (scRNA) typically requires manual labor but this process is time-consuming and requires the labeler to have a comprehensive background in biology. Many machine learning methods recently developed specialize in automated label generation for single-cell data types. Here, we implemented a machine learning method that utilizes a variational autoencoder (VAE) and random forest model to predict cell types (B, CD4 T, CD8 T, DC, Mono, NK, other, and other T) in bone marrow aspirate concentrate (BMAC)

## Data Availability

The BMAC data we used for training and testing is from the New York Genome Center, as part of their integrated analysis of multimodal single-cell data, and can be downloaded [here](https://atlas.fredhutch.org/nygc/multimodal-pbmc/). We downloaded the full (raw) dataset of 161,159 cells. This dataset has three levels of labeling (L1, L2, and L3) along with information on the gene expressions across cells. L1 is the broadest level and has 8 categories: B, CD4 T, CD8 T, DC, Mono, NK, other, and other T. L3 is the more specific, with 57 categories. We focused on L1 for our model and aimed to be able to classify cells across 8 categories accurately. 
