
import numpy as np
import os as os
import re

source_file = 'treasure-island.txt'
output_file = 'embeddings.csv'

class Vocabulary:
  '''A vocabulary is a list of possible input values, plus methods to 
     translate them from human readable format to 1-of-k vectors.'''
  def __init__(self, values):
    self._vocab = values
    self._vocab_size = len(values)
    self._vocab_index_lut = { val:index for index,val in enumerate(self._vocab) }

  def value_to_vector(self, value):
    '''Converts a value to a 1-of-k vector.'''
    output = np.zeros((self._vocab_size, 1), dtype=np.float64)
    output[self._vocab_index_lut[value]] = 1
    return output
  
  def index_to_value(self, index):
    return self._vocab[index]

  def vector_to_value(self, vector):
    '''Converts a 1-of-k vector to a value (character/word).'''
    max_index = vector.argmax()
    return self._vocab[max_index]

  def vocab_size(self):
    '''Returns the number of elements in our vocabulary.'''
    return len(self._vocab)

class DataSet:
  '''A dataset consists of a set of data samples and ground truth.'''
  def __init__(self, path):
    '''Loads a text file and configures the dataset.'''
    self.source_data = []
    self.ground_truth = []
    tokens = set()

    print("Parsing file: " + path)
    contents = open(path, 'r').read()
    contents = re.findall(r"[\w']+|[.,!?;]", contents)
    tokens.update(contents)
    self.source_data.extend(contents[:len(contents)-1])
    self.ground_truth.extend(contents[1:])
    self.vocab = Vocabulary(list(tokens))

    print("Our vocabulary contains " + str(self.vocab.vocab_size()) + " tokens.")

  def sample(self, start, count):
    '''Returns a tuple of samples within the range specified.'''
    if start + count > len(self.source_data):
      count = len(self.source_data) - start
    return (self.source_data[start:start+count], self.ground_truth[start:start+count])

class RecurrentNetwork:
  learning_rate = 1e-1
  batch_size = 25
  epoch_count = 2

  def __init__(self, vocab, hidden_size):
    '''Initializes our shape.'''
    self._input_size = vocab.vocab_size()
    self._hidden_size = hidden_size
    self._output_size = vocab.vocab_size()
    self.vocab = vocab
    self.clear()

  def clear(self):
    '''Clears the state of our model.'''
    self._Wxh = np.random.randn(self._hidden_size, self._input_size).astype(np.float64) * 0.01
    self._Whh = np.random.randn(self._hidden_size, self._hidden_size).astype(np.float64) * 0.01
    self._Why = np.random.randn(self._output_size, self._hidden_size).astype(np.float64) * 0.01
    self._bh = np.zeros((self._hidden_size, 1), dtype=np.float64)
    self._by = np.zeros((self._output_size, 1), dtype=np.float64)

  def feed_forward(self, x, hp):
    '''Feeds one input vector through our model and returns the updated state and probabilities.'''
    h = np.tanh(np.dot(self._Wxh, x) + np.dot(self._Whh, hp) + self._bh)
    y = np.dot(self._Why, h) + self._by
    p = np.exp(y) / np.sum(np.exp(y))
    return (h, y, p)
    
  def SGD(self, input, truth, hidden_state):
    '''Stochastic gradient descent.'''
    hd, yd, pd = {}, {}, {}
    # Deltas for our weights and biases.
    deltaWxh = np.zeros_like(self._Wxh)
    deltaWhh = np.zeros_like(self._Whh)
    deltaWhy = np.zeros_like(self._Why)
    deltabh = np.zeros_like(self._bh)
    deltaby = np.zeros_like(self._by)
    deltahprime = np.zeros_like(hidden_state)

    # Feed forward our batch samples through the network.
    total_loss = 0
    hd[-1] = np.copy(hidden_state)
    for t in range(len(input)):
      hd[t], yd[t], pd[t] = self.feed_forward(input[t], hd[t-1])
      total_loss += -pd[t][truth[t].argmax()] + 1.0

    # Back propagate to capture our delta gradient.
    for t in reversed(range(len(input))):
      deltay = np.copy(pd[t])
      deltay[truth[t].argmax()] -= 1
      deltaWhy += np.dot(deltay, hd[t].T)
      deltaby += deltay
      deltah = np.dot(self._Why.T, deltay) + deltahprime
      dhpre = (1 - hd[t] * hd[t]) * deltah
      deltabh += dhpre
      deltaWxh += np.dot(dhpre, input[t].T)
      deltaWhh += np.dot(dhpre, hd[t-1].T)
      deltahprime = np.dot(self._Whh.T, dhpre)

    return (total_loss, hd[len(input)-1], deltaWxh, deltaWhh, deltaWhy, deltabh, deltaby)

  def train(self, dataset):
    '''Trains our network using the dataset provided.'''
    batches_per_epoch = int(len(dataset.source_data) / self.batch_size + 0.5)
    batch_loss = 0
    self.clear()

    # Memory for adaptive gradient. Especially necessary with longer BPTT.
    memWxh = np.zeros_like(self._Wxh)
    memWhh = np.zeros_like(self._Whh)
    memWhy = np.zeros_like(self._Why)
    membh = np.zeros_like(self._bh)
    memby = np.zeros_like(self._by)

    for epoch in range(self.epoch_count):
      hidden_state = np.zeros((self._hidden_size,1), dtype=np.float64)
      for batch in range(batches_per_epoch):
        (input, truth) = dataset.sample(batch * self.batch_size, self.batch_size)
        input = [dataset.vocab.value_to_vector(x) for x in input]
        truth = [dataset.vocab.value_to_vector(x) for x in truth]
        loss, hidden_state, deltaWxh, deltaWhh, deltaWhy, deltabh, deltaby = self.SGD(input, truth, hidden_state)
        batch_loss += loss
        # Apply gradient update using adaptive gradient process (adagrad).
        for param, delta, memory in zip([self._Wxh, self._Whh, self._Why, self._bh, self._by], 
                                        [deltaWxh, deltaWhh, deltaWhy, deltabh, deltaby],
                                        [memWxh, memWhh, memWhy, membh, memby]):
          delta = np.clip(delta, -5, 5)
          memory += delta * delta
          param += -self.learning_rate * delta / np.sqrt(memory + 1e-8)

  def get_embedding(self):
    return self._Wxh

def export_embedding(path, vocab, embedding):
  file = open(path, 'a')
  for index in range(vocab.vocab_size()):
    file.write(vocab.index_to_value(index) + ', ')
    value = embedding[:, index]
    for vindex in range(value.shape[0]):
      file.write(str(value[vindex]))
      if vindex < value.shape[0] - 1:
        file.write(", ")
    file.write('\n')
  file.close()

data = DataSet(source_file)
print("Training the network (this may take a bit...).")
network = RecurrentNetwork(data.vocab, 8)
network.train(data)
print("Exporting the embedding to file: %s." % output_file)
embedding = network.get_embedding()
export_embedding(output_file, data.vocab, embedding)