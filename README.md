# Patch-based Whole Slide Image Classification with Advanced Selection Methods and Feature Extraction

## Overview

This repository contains the implementation of a novel approach for classifying whole slide images (WSIs) in digital pathology using patch-based methods with advanced selection techniques and feature extraction. The project focuses on improving the accuracy and efficiency of WSI classification by optimizing patch selection strategies and feature extraction methods.

## Background

Digital pathology has emerged as a critical field in medical imaging, with whole slide images (WSIs) being the primary data format. Due to the extremely high resolution of WSIs (often gigapixels in size), direct classification is computationally challenging. Patch-based approaches, which divide WSIs into smaller, manageable patches for analysis, have become the standard methodology. However, these approaches face challenges in:

1. Selecting the most informative patches from thousands of candidates
2. Extracting meaningful features from these patches
3. Aggregating patch-level information to make slide-level predictions

This project addresses these challenges through advanced selection methods and feature extraction techniques.

## Features

- **Patch Extraction**: Abilities for extracting patches from whole slide images
- **Patch Cleaning**: Methods to filter out non-informative patches (e.g., background, artifacts)
- **Advanced Selection Methods**: Algorithms to identify the most diagnostically relevant patches
- **Feature Extraction**: Techniques to extract meaningful features from selected patches
- **Classification Models**: Implementation of multiple classifiers (MLP, Random Forest, XGBoost)
- **Comparison Framework**: Abilities to compare different methodologies and models

## Repository Structure

- `clean_patches.py`: Script for filtering and cleaning extracted patches
- `compare.py`: Framework for comparing different models and methodologies
- `create_patches_directories.py`: Utility for organizing patches into appropriate directory structures
- `mlp.py`: Implementation of Multi-Layer Perceptron classifier
- `patch_dataset.py`: Dataset handling for patch-based classification
- `rf.py`: Implementation of Random Forest classifier
- `xgboost.py`: Implementation of XGBoost classifier
