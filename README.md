# Word2Vec Implementation in PyTorch

## Overview

This repository contains an implementation of the **Word2Vec** algorithm using **PyTorch** to learn word embeddings. The model is trained on a sample text paragraph, and the embeddings are learned using the **Skip-gram** approach. The goal is to create continuous vector representations for words where semantically similar words are closer in the vector space.

The Word2Vec model consists of two key components:
- **Encoder (Input Layer)**: Takes the context words as input.
- **Decoder (Output Layer)**: Predicts the target word based on the context.

### Key Features

- **Word Embeddings**: Learn dense vector representations for words using the Skip-gram model.
- **Cosine Similarity**: Find the most similar words to a given word using cosine similarity.
- **3D Visualization**: Visualize the learned word embeddings in a 3D space using PCA (Principal Component Analysis).
- **PyTorch Model**: Built with PyTorch for efficient training and flexible use in deep learning applications.

## Dataset

For the purpose of this implementation, a sample paragraph is used. The text is tokenized, preprocessed (lowercase and stopword removal), and context-target pairs are created for training the model. 

Example text used for training: You can use any text corpus in place of the "paragraph"


## Model Architecture

The model is a simple neural network with the following layers:

1. **Embedding Layer**: Converts the input words into vector representations (word embeddings).
2. **Linear Layer**: Passes the embeddings through a linear layer to predict the target word in the context.

The model utilizes the **Skip-gram** approach, which predicts context words based on the target word.

## Training

The model is trained over multiple epochs with the following configuration:

- **Batch Size**: 8
- **Optimizer**: Adam optimizer with a learning rate of 0.001
- **Loss Function**: CrossEntropyLoss

During training, context-target pairs are used to adjust the embeddings, minimizing the loss function. The loss value is printed at the end of each epoch.

## 3D Visualization

The learned word embeddings are visualized using **Principal Component Analysis (PCA)** to reduce the embeddings to 3 dimensions. A 3D scatter plot is generated, where each point represents a word, and the distance between points indicates the similarity between the words.

## Finding Similar Words

After training, the model allows querying for the most similar words to a given word. This is done by computing **cosine similarity** between the embedding of the input word and all other embeddings in the vocabulary.

For example, you can input the word "algorithm" and get the most similar words based on the learned embeddings.

## License

This project is licensed under the **MIT License**.


