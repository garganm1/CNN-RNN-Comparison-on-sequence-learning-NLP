![cnn](https://user-images.githubusercontent.com/55252306/109691209-35cd0f80-7b55-11eb-810d-c2861dae1701.PNG)
# 1D-CNN-Bi-LSTM-Comparison-on-sequence-learning-NLP

This repo deals with sequence learning of articles using CNN and RNN architectures. The dataset consists of article content, their titles, author names, date of publishing, and the name of publishers. The articles' content will be processed and analyzed to predict the name of the publisher. This would help us tell if the way of writing by publishers is unique to their own as the sequences of articles' content will be learned by the models.

CNN usually deals with convolutions of tensors such that it tries to find meaningful features in a localized manner depending on kernel size. In this repo, instead of localized transformations, a 1-D CNN will be operated on the sequence of tokenized text to generate feature transformations that learn the sequences within the articles. In contrast, RNN architectures are mainly built to learn the sequences and so, in this repo, a comparison will be made between the performances of a CNN vs RNN architecture on sequence learning.

There are three essential notebooks here-

1. Part 1 deals with cleaning and processing of the data to build the models
2. In Part 2, the data is tokenized and then a 1-dimensional CNN architecture will be employed to train and test the model
3. In Part 3, a Bi-directional LSTM network will be implemented

## Part 1: Data Cleaning

There are simple cleaning procedures implemented in this notebook. The mail addresses and usernames, if found, are removed. All punctuations, stopwords are removed. The words are lower-cased and finally lemmatization and stemming (with Snowball Stemmer) are applied.

As there are several publishers (as target), the slicing is performed to get the articles of only the top 5 publishers (by count) viz. 


| Label   | Publisher Name |
| ------------- | ------------- |
| 0  | Breitbart  |
| 1  | New York Post  |
| 2  | NPR (National Public Radio)  |
| 3  | Washington Post  |
| 4  | Reuters  |

Finally, the target i.e. the name of the publisher is one-hot encoded.

After all of the processing, a 70-30 split is made to get the training set and testing set with a stratified target (train+test dataset = 74548 data points)

## Part 2: Data Tokenization and 1-D CNN model employed

The tokenizer and padding functionalities are implemented first to tokenize the words of the articles in the training dataset. The tokenizer is first fit onto the training set that assigns a number to each unique word. Each article will then have an array of numbers that denote a particular word stored in the tokenizer's word-index dictionary.

As can be assumed, articles' length will vary and so the number of columns would be equal to the maximum length of all articles in the dataset. For the articles with shorter lengths, zeros will be padded after their length is finished to denote that no words are present at their positions.

For example, for the below corpus-

A1 :- I am here. <br>
A2 :- He is here now.

The tokenized and padded form will be-

A1 :- [1, 2, 3, 0] <br>
A2 :- [4, 5, 3, 6]

The tokenizer fit on the training set will then encode the testing set as per its word-index dictionary taken from the training set. There are certain dataset parameters that need to be fed to the model viz. maximum length of all articles and the vocabulary size of the training set.

#### CNN Architecture

The model takes three layers as inputs (matrices of encoded vectors of the articles just repeated thrice). Each layer is evaluated as a separate channel with a different kernel_size that traverses and convolutes with the vectors.

An embedding layer first embeds the encoded vectors to generate another vector of size 10 for each word. This process is applied to all three channels.

A 1-dimensional convolutional layer then traverses over the matrix of embedded vectors (meaning the stride happens only in the vertical direction - see figure below denoting red boundary box as the kernel). The kernel size differs for the three channels producing features that capture the sequence of words with different lengths.

A dropout, 1-D pooling and flatten layer is applied to each channel with the same hyper-parameters applied across the three channels.

The flattened features from the three layers are then concatenated together to produce a single set of features which is then passed on to another dense layer that finally gives out an output containing 5 neurons for each label.

![image.png](attachment:cnn.png)
