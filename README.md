# NLD - Natural Language Decorators

This is a package to carry out common text preprocessing tasks in NLTK using dedicated decorators from a class that can also help keep track of the preprocessing steps taken, time it took preprocessing and build simple pipelines faster, especially for when simple exploratory analysis is being carried out.


## Install

To install the package, from the root directory of the repository, run the following:

`python3 setup.py install --user`

This will also install nltk 3.4.5

## Examples

The followings are some examples of the preprocessing steps that can be applied, including mistakes that may go undetected.

```python
from nltk.tokenize import word_tokenize

# Jane Austen book Emma obtained from the Gutenberg Project, which is for some weird reason banned in Italy.
with open("~/Documents/austen.txt") as raw_text:
    raw_text = raw_text.read()

# Instantiate the NLD object, you can set the logger on and store all the timings for each run if you want
nldecorator = nld.NLD(logger=True, store_all_process_times=True)

@nldecorator.timeit
@nldecorator.freq_dist()
@nldecorator.stem
@nldecorator.remove_stopwords(punct=True) # this key argument must be specified
@nldecorator.substitute([("emma", "peppa")])
@nldecorator.lower
def tokenize(_input: str) -> list:
    # This single function with a pipeline of decorators is a single run
    return word_tokenize(_input)

first_result = tokenize(raw_text)

print("1\n", first_result, "\n")
print("PROCESS TIME:", nldecorator.process_time)
print("Run ID: ", nldecorator.id)
print("Decorators used:", nldecorator.chain[nldecorator.id])
```
```
1
 [('mr.', 1080), ('peppa', 860), ('could', 834), ("'s", 831), ('would', 815)] 

PROCESS TIME: 2.593135118484497
Run ID:  2d1c614f-e5ba-41d3-aea7-ffab8e5e7a7b
Decorators used: tokenize-lower_wrapper-sub_wrapper-rm_stopwords_wrapper-stem_wrapper-freq_dist_wrapper-
```


```python
# Wrong Order, stopwords with capitals still present
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
print("Run ID: ", nldecorator.id)
print("Decorators used:", nldecorator.chain[nldecorator.id])
```

```
3
 [('\ufeffthe', 'project', 'gutenberg'), ('project', 'gutenberg', 'ebook'), ('gutenberg', 'ebook', 'emma'), ('ebook', 'emma', ','), ('emma', ',', 'jane'), (',', 'jane', 'austen'), ('jane', 'austen', 'this'), ('austen', 'this', 'ebook'), ('this', 'ebook', 'use'), ('ebook', 'use', 'anyon'), ('use', 'anyon', 'anywher'), ('anyon', 'anywher', 'cost'), ('anywher', 'cost', 'almost'), ('cost', 'almost', 'restrict'), ('almost', 'restrict', 'whatsoev'), ('restrict', 'whatsoev', '.'), ('whatsoev', '.', 'you'), ('.', 'you', 'may'), ('you', 'may', 'copi'), ('may', 'copi', ',')] 

PROCESS TIME: 2.7864768505096436
Run ID:  6f42acfa-0407-4ddd-8f91-40309db5f08b
Decorators used: tokenize-rm_stopwords_wrapper-lower_wrapper-stem_wrapper-ngrams_wrapper-
```

The following is an example using the `itarator` decorator and then the `open_from_path` decorator  

```python
nldecorator = nld.NLD(logger=True, store_all_process_times=True)

# With the open_from_path decorator you can run through the pipeline all the files from a given directory or a single file

@nldecorator.timeit
@nldecorator.lower
@nldecorator.word_tokenizer
@nldecorator.iterator()
@nldecorator.open_from_path
def return_directory():
    # there are three files of the same book in this directory.
    return "~/Documents/books/"

return_directory()
print(nldecorator.all_process_times)
return_directory()
print(nldecorator.all_process_times)
return_directory()
print(nldecorator.all_process_times)
```

```
[('lower_wrapper', 1.363213062286377)]
[('lower_wrapper', 1.363213062286377), ('lower_wrapper', 1.3505115509033203)]
[('lower_wrapper', 1.363213062286377), ('lower_wrapper', 1.3505115509033203), ('lower_wrapper', 1.2218332290649414)]
```

### Build Dataframes

```python
@nldecorator.build_df(column="tags")
@nldecorator.pos_tagger
@nldecorator.iterator()
def preprocess_tags(sents):
    return sents

@nldecorator.build_df("tokens")
@nldecorator.stem
@nldecorator.remove_stopwords()
@nldecorator.word_tokenizer
@nldecorator.lower
@nldecorator.iterator()
def preprocess_tokens_iter(sents):
    return sents


sents = ["This one is my awesome string, written by myself personally.", 
         "This two is my awesome string, written by myself personally 2.",
         "This three is my awesome string, written by myself personally 3."]

for i in range(3):
    preprocess_tokens_iter(sents)
    preprocess_tags(sents)

nldecorator.df
```


| 	| tokens | 	tags |
| --- | ------ | ----- |
| 0 	| [one, awesom, string, ,, written, person, .] |    [(This, DT), (one, CD), (is, VBZ), (my, PRP$),... |
| 1 	| [two, awesom, string, ,, written, person, 2, .] | [(This, DT), (two, CD), (is, VBZ), (my, PRP$),... |
| 2 	| [three, awesom, string, ,, written, person, 3, .] |   [(This, DT), (three, CD), (is, VBZ), (my, PRP$... |
