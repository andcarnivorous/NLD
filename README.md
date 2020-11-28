# NLD - Natural Language Decorators

This is a package to carry out common text preprocessing tasks in NLTK using dedicated decorators from a class that can also help keep track of the preprocessing steps taken, time it took preprocessing and build simple pipelines faster, especially for when simple exploratory analysis is being carried out.


The followings are some examples of the preprocessing steps that can be applied, including mistakes that may go undetected.

```python
from nld import nld
from nltk.tokenize import word_tokenize

nldecorator = nld.NLD(logger=True, store_all_process_times=True)

# Get the most frequent words after removing stopwords, applying stemming and substituting Emma with Peppa.
@nldecorator.timeit
@nldecorator.freq_dist
@nldecorator.stem
@nldecorator.remove_stopwords(True)
@nldecorator.substitute([("emma", "peppa")])
@nldecorator.lower
def tokenize(_input):
    return word_tokenize(_input)
first_result = tokenize(raw_text)
print("1\n", first_result, "\n")
print("PROCESS TIME:", nldecorator.process_time)
print(nldecorator.id)

# Get the most frequent bigrams after stemming and substituting Emma with Peppa.
@nldecorator.timeit
@nldecorator.freq_dist
@nldecorator.n_grams(2)
@nldecorator.stem
@nldecorator.substitute([("emma", "peppa")])
@nldecorator.lower
def tokenize(_input):
    return word_tokenize(_input)
second_result = tokenize(raw_text)
print("2\n", second_result[:20], "\n")
print("PROCESS TIME:", nldecorator.process_time)
print(nldecorator.id)

# Wrong Order, stopwords with capitals still present because decorator `lower` is used after `remove_stopwords`.
@nldecorator.timeit
@nldecorator.n_grams(3)
@nldecorator.stem
@nldecorator.lower
@nldecorator.remove_stopwords()
def tokenize(_input):
    return word_tokenize(_input)

third_result = tokenize(raw_text[:])

print("3\n", list(third_result)[:20], "\n")
print("PROCESS TIME:", nldecorator.process_time)
print(nldecorator.id)

# Get named entities after removing stopwords and tagging.
@nldecorator.timeit
@nldecorator.named_entity
@nldecorator.pos_tagger
@nldecorator.remove_stopwords()
def tokenize(_input):
    return word_tokenize(_input)

fourth_result = tokenize(raw_text[:])
print()
print("4\n", list(fourth_result)[:20], "\n")
print("PROCESS TIME:", nldecorator.process_time)
```

The following is an example using the `itarator` decorator and then the `open_from_path` decorator  

```python
@nldecorator.timeit
@nldecorator.stem
@nldecorator.lower
@nldecorator.word_tokenizer
@nldecorator.iterator
def return_strings_list(sents):
    return sents

sents = [
    "This one is my awesome string, written by myself personally 1.",
    "This is my awesome string, written by myself personally 2.",
    "This is my awesome string, written by myself personally 3."]

# At each call the next item from `sents` will go through the decorators after `iterable`
print(return_strings_list(sents))
print(return_strings_list(sents))
print(return_strings_list(sents))


@nldecorator.lower
@nldecorator.word_tokenizer
@nldecorator.iterator
@nldecorator.open_from_path
def process_from_this_dir():
    dir = "/home/main/Documents/books/"
    return dir

# This will get a list of files in the books directory.
# and then pass at each call the next file to the other decorators.
process_from_this_dir()
```

## Install

To install the package, from the root directory of the repository, run the following:

`python3 setup.py install --user`

This will also install nltk 3.4.5