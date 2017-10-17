# Word2vec
Sentiment analysis using deep learning (word2vec model).

Two word2vec algorithms implemented: skip-gram (using negative sampling, for efficiency) and CBOW (continuous bag of words). 

Useful link to an explanation for the skip-gram model (and in general, for the word2vec model): http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

After running word2vec.py, we have vector representations for each word in our dataset (these vectors encapsulate the whole meaning of the words in the corpus and their values will cluster based on similarities between words - an example is shown in word_vectors_visualization.png). These word-vectors are then used to perform the (very simple) sentiment analysis task. 

To train the word vectors we used the Stanford Sentiment Treebank (SST) dataset. Run sh get datasets.sh to get the dataset. Stochastic gradient descent is used for updating. The entire training process will take quite a bit of time (anywhere between 2 and 3 hours, it's performing 40,000 iterations).

Note: No Tensorflow or any other machine learning libraries used. All gradients and cost functions are implemented from scratch, based on Stanford's Deep Learning course. There are various "sanity-check" tests for all important implemented functionalities.
