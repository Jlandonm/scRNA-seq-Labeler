# scRNA-seq-Labeler

## Overview

The exponentially increasing amount of RNA sequencing data following the development of RNA sequencing technology provides novel knowledge about high-resolution profiling of cell transcriptomes. However, one of the largest challenges with RNA-seq data is properly labeling data for analysis. Labeling the cell clusters of single-cell RNA-seq data (scRNA) typically requires manual labor but this process is time-consuming and requires the labeler to have a comprehensive background in biology. Many machine learning methods recently developed specialize in automated label generation for single-cell data types. Here, we implemented a machine learning method that utilizes a variational autoencoder (VAE) and random forest model to predict cell types (B, CD4 T, CD8 T, DC, Mono, NK, other, and other T) in bone marrow aspirate concentrate (BMAC)
