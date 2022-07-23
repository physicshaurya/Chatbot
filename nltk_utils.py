import nltk
import numpy as np 

# nltk.download('punkt')

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

#Tokenizing the sentence 
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

#Stemming the word
def stem(word):
    return stemmer.stem(word.lower())

#Testing Tokenizing

# a = "How are you ?"

# print(a)

# a = tokenize(a)

# print(a)

#Testing Stemming

# words = ["organize", "organizes", "organizing"]

# stemmed_words = [stem(word) for word in words]

# print(stemmed_words)

def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag