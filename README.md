# SentimentNet

A sentiment analysis project implementing Yoon Kim's paper on **Convolutional Neural Networks for Sentence Classification**. This model classifies movie reviews as positive or negative using pre-trained GloVe embeddings and a multichannel CNN architecture.

## Overview
This project demonstrates how convolutional neural networks (CNNs) can be used for text classification tasks. It leverages pre-trained word embeddings to capture semantic meaning and improve classification accuracy. The model has been implemented to classify sentiment in movie reviews, similar to the approach detailed in Yoon Kim's 2014 paper.

## Features
- **Pre-trained GloVe Embeddings**: Enhances the model’s understanding of word relationships.
- **Multichannel CNN Architecture**: Combines both static and fine-tuned embeddings.
- **Real-time Sentiment Prediction**: Classifies new text inputs with high accuracy.
- **Easily Reproducible**: The codebase includes scripts for data preprocessing, model training, and evaluation.

## Requirements
- Python 3.7+
- TensorFlow 2.0+
- NumPy
- TensorFlow Datasets
- GloVe Embeddings (`glove.6B.300d.txt`)
- Additional packages specified in `requirements.txt`

## Architecture
The model architecture follows the approach in Yoon Kim's paper:

- **Embedding Layer**: Uses pre-trained GloVe embeddings (static and non-static versions).
- **Conv1D Layers** : Applies filters to extract features from n-grams (3-gram, 4-gram, 5-gram).
- **Max Pooling**: Captures the most important features.
- **Dense Layer**: Predicts the sentiment using a softmax output.

## References
- [Yoon Kim's 2014 paper](https://arxiv.org/abs/1408.5882)
- [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/)
