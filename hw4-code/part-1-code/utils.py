import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###
    qwerty_neighbors = {        
        'q': ['w', 'a'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'], 
        'r': ['e', 't', 'f'], 't': ['r', 'y', 'g'], 'y': ['t', 'u', 'h'],
        'u': ['y', 'i', 'j'], 'i': ['u', 'o', 'k'], 'o': ['i', 'p', 'l'],
        'p': ['o', 'l'],
        'a': ['q', 's', 'z'], 's': ['w', 'a', 'd', 'z', 'x'], 
        'd': ['e', 's', 'f', 'x', 'c'], 'f': ['r', 'd', 'g', 'c', 'v'],
        'g': ['t', 'f', 'h', 'v', 'b'], 'h': ['y', 'g', 'j', 'b', 'n'],
        'j': ['u', 'h', 'k', 'n', 'm'], 'k': ['i', 'j', 'l', 'm'],
        'l': ['o', 'k', 'p'],
        'z': ['a', 's', 'x'], 'x': ['s', 'd', 'z', 'c'], 
        'c': ['d', 'f', 'x', 'v'], 'v': ['f', 'g', 'c', 'b'],
        'b': ['g', 'h', 'v', 'n'], 'n': ['h', 'j', 'b', 'm'],
        'm': ['j', 'k', 'n']
    }
    text = example["text"]
    words = text.split()
    transformed_words = []
    word_p = 0.15
    char_p = 0.3

    for word in words:
        if random.random() < word_p:
            word_chars = list(word)
            for i, char in enumerate(word_chars):
                lower_char = char.lower()
                if lower_char in qwerty_neighbors and random.random() < char_p:
                    neighbor = random.choice(qwerty_neighbors[lower_char])
                    if char.isupper():
                        neighbor = neighbor.upper()
                    word_chars[i] = neighbor
            transformed_words.append(''.join(word_chars))
        else:
            transformed_words.append(word)

    example["text"] = " ".join(transformed_words)
    return example