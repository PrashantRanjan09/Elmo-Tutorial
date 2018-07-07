# Elmo-Tutorial

This is a short tutorial on using Deep contextualized word representations (ELMo) which is discussed in the paper https://arxiv.org/abs/1802.05365.
This tutorial can help in using:

**Pre Trained Elmo Model** <br>
**Training an Elmo Model on your new data** <br>
**Incremental Learning** <br>
while doing Incremental training make sure to put your embedding layer name in line 758:

    exclude = ['the embedding layer name yo want to remove']

Visualization of the word vectors using Elmo:

* Tsne
* Tensorboard 

### Using Elmo Embedding layer in consequent models
if you want to use Elmo Embedding layer in consequent model build refer : https://github.com/PrashantRanjan09/WordEmbeddings-Elmo-Fasttext-Word2Vec
