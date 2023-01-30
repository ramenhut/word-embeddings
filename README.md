# Simple word embedding vectorizer
A simple recurrent neural network that generates word embeddings given a training text file. Neural networks prefer dense low magnitude tensors. Word embeddings are numerical representations of words in a vector space that capture semantic meaning through proximity of the vectors. They are used in NLP tasks such as language modeling, text classification, and others.

We typically generate word embeddings using a neural network that's trained to satisfy some objective (e.g. predict the next word in a sequence, categorize words according to some criteria).
We can then evaluate the quality of the embedding space by examining the clustering of words that are commonly used together, or have similar meaning.

For this project we've used a shallow network that's trained to predict the next word in a sequence, given a text file as training data. Once the network is trained,
we extract the input-to-hidden layer weights to use them as our embedding, and save them to a comma-separated-values file. Our embedding is a 2D tensor of size WxV, where W is a configurable embedding tensor size, and V is the number of words in our vocabulary (sampled from the training data).