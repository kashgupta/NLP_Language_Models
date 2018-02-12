from collections import *
from random import random
import math

def train_char_lm(fname, order=4, add_k=1):
  ''' Trains a language model.

  This code was borrowed from 
  http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139

  Inputs:
    fname: Path to a text corpus.
    order: The length of the n-grams.
    add_k: k value for add-k smoothing. NOT YET IMPLMENTED

  Returns:
    A dictionary mapping from n-grams of length n to a list of tuples.
    Each tuple consists of a possible net character and its probability.
  '''

  data = open(fname).read()
  lm = defaultdict(Counter)
  pad = "~" * order
  data = pad + data
  add_k = float(add_k)
  possible_char = set()
  for i in range(len(data)-order):
    history, char = data[i:i+order], data[i+order]
    lm[history][char]+=1
    possible_char.add(char)
  def normalize(counter):
    V = len(possible_char)
    s = float(sum(counter.values())) + V*float(add_k)
    return [(c, (counter[c]+add_k)/s) if c in counter else (c, add_k/s) for c in possible_char]
  outlm = {hist:normalize(chars) for hist, chars in lm.items()}
  return outlm


def generate_letter(lm, history, order):
  ''' Randomly chooses the next letter using the language model.
  
  Inputs:
    lm: The output from calling train_char_lm.
    history: A sequence of text at least 'order' long.
    order: The length of the n-grams in the language model.
    
  Returns: 
    A letter
  '''
  
  history = history[-order:]
  dist = lm[history]
  x = random()
  for c,v in dist:
    x = x - v
    if x <= 0: return c
    
    
def generate_text(lm, order, nletters=500):
  '''Generates a bunch of random text based on the language model.
  
  Inputs:
  lm: The output from calling train_char_lm.
  history: A sequence of previous text.
  order: The length of the n-grams in the language model.
  
  Returns: 
    A letter  
  '''
  history = "~" * order
  out = []
  for i in range(nletters):
    c = generate_letter(lm, history, order)
    history = history[-order:] + c
    out.append(c)
  return "".join(out)

def perplexity(test_filename, lm, order=4):
  '''Computes the perplexity of a text file given the language model.
  
  Inputs:
    test_filename: path to text file
    lm: The output from calling train_char_lm.
    order: The length of the n-grams in the language model.
  '''
  test = open(test_filename).read()
  pad = "~" * order
  test = pad + test
  log_sum = 0
  N = 0
  for i in range(len(test)-order):
    history, char = test[i:i+order], test[i+order]
    log_sum += float(1)/math.log(lm[history][char])
    N+=1

  return math.exp(log_sum)**(1.0/N)


def calculate_prob_with_backoff(char, history, lms, lambdas):
  '''Uses interpolation to compute the probability of char given a series of 
     language models trained with different length n-grams.

   Inputs:
     char: Character to compute the probability of.
     history: A sequence of previous text.
     lms: A list of language models, outputted by calling train_char_lm.
     lambdas: A list of weights for each lambda model. These should sum to 1.
    
  Returns:
    Probability of char appearing next in the sequence.
  '''
  return sum(lambdas(i)*lms(i)[history][char] for i in len(lambdas))


def set_lambdas(lms, dev_filename):
  '''Returns a list of lambda values that weight the contribution of each n-gram model

  This can either be done heuristically or by using a development set.

  Inputs:
    lms: A list of language models, outputted by calling train_char_lm.
    dev_filename: Path to a development text file to optionally use for tuning the lmabdas. 

  Returns:
    Probability of char appearing next in the sequence.
  '''
  return [1.0/len(lms) for i in len(lms)]

if __name__ == '__main__':
  print('Training language model')
  lm = train_char_lm("shakespeare_input.txt", order=2)
  print(generate_text(lm, 2))
