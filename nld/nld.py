import logging
import string
import uuid
from time import time
import pandas as pd

from nltk import FreqDist
from nltk import ngrams, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize

from .utils import *

LANGUAGES = stopwords.fileids()


class NLD(object):
    """
    The NLD object contains the NLD decorators. The `stopwords`, `store_all_process_times`, `iterables` and `logger` attributes are
    used in the decorators. The other attributes are used to keep track of each run and the decorators used.
    """
    def __init__(self, logger=None, store_all_process_times=False, language="english"):
        self.__name__ = "CompLing"
        if logger:
            logging.basicConfig(
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.INFO)
            self.logger = logging.getLogger(self.__name__)
        else:
            self.logger = None
        self.process_time = None
        self.language = language
        try:
            self.stopwords = list(set(stopwords.words(self.language))) + ["”", '--', '“']
        except LookupError:
            raise LookupError("You miss the stopwords module from NLTK, which is required for NLD. Execute nltk.download('stopwords') to download it")
        self.store_all_process_times = store_all_process_times
        self.all_process_times = dict()
        self.chain = dict()
        self.iterable = dict()
        self.id = None
        self.ids = []
        self.no_input = False
        self.df = None

    def build_series(self, _func=None, *, vals=None):
        """
        Creates a series out of a list output from the previous function. If this is a list of lists or list of tuples,
        passing the value 'word' to the parameter `vals` will get only the tokens from the output.
        Passing `output` to `vals` will instead return only the second item in the results.
        :param vals: a string that should be either "word" or "output"
        """
        if vals and vals.lower() not in ["word", "output"]:
            raise ValueError("vals must be either `word` or `output` if provided")

        @nldmethod
        def build_series_decorator(func):
            self._check_id(func)
            self.chain[self.id]  += func.__name__ + "-"

            @nldmethod
            def build_series_wrapper(_input=None):
                result = func(_input) if _input else func()
                if vals:
                    result = pd.Series([x[0 if vals == "word" else 1] for x in result])
                else:
                    result = pd.Series(result)
                return result
            return build_series_wrapper
        if not _func:
            return build_series_decorator
        else:
            return build_series_decorator(_func)

    def build_df(self, column, category=None):
        """
        Returns a NLTK FreqDist from a given list.
        :param number: Number of top most frequent items
        """
        @nldmethod
        def build_df_decorator(func):
            self._check_id(func)
            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def build_df_wrapper(_input=None):
                if isinstance(self.df, type(None)):
                    self.df = pd.DataFrame()
                if column not in self.df.columns:
                    self.df[column] = None
                    if category is not None and "class" not in self.df.columns:
                        self.df["class"] = None
                    if self.logger: self.logger.info("Build DF : Created column: %s", column)
                result = func(_input) if _input else func()
                new_row = self.df[column].count()
                if not isinstance(column, pd.Series):
                    self.df.loc[new_row, column] = result
                else:
                    self.df[column] = result
                if category is not None:
                    self.df.loc[new_row, "class"] = category
                return result

            @nldmethod
            def build_df_from_series_wrapper(_input=None):
                if isinstance(self.df, type(None)):
                    self.df = pd.DataFrame()
                if column not in self.df.columns:
                    self.df[column] = None
                    if category is not None and "class" not in self.df.columns:
                        self.df["class"] = None
                    if self.logger: self.logger.info("Build DF : Created column: %s", column)
                result = func(_input) if _input else func()
                self.df[column] = result
                return result

            if func.__name__ in ["build_series", "build_series_decorator", "build_series_wrapper"]:
                return build_df_from_series_wrapper
            return build_df_wrapper
        return build_df_decorator

    def add_stopwords(self, new_stopwords):
        """Adds stopwords to the NLD attribute stopwords."""
        if isinstance(new_stopwords, str):
            new_stopwords = [new_stopwords]
        self.stopwords += new_stopwords

    def set_logger_level(self, level):
        """
        Sets the logger, if active, on the given level
        :param level: string of the level to set the logger, can be: info, warning, error
        :return:
        """
        if not self.logger:
            raise AttributeError("Logger not set, please set the logger attribute to True before setting the logger level")
        level = level.lower()
        levels = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
        if level in levels:
            self.logger.level = levels[level]
        else:
            raise KeyError("Level provided is incorrect, available levels are ", levels.keys())

    def _check_id(self, func):
        """
        Sets the id of the current run, if the given function is not a NLD decorator it will create an id for a new run.
        :param func: a function
        :return:
        """
        if not self.id:
            self.id = str(uuid.uuid4())
            self.chain[self.id] = ""
            self.ids.append(self.id)
        elif not hasattr(func, "nldmethod"):
            self.id = str(uuid.uuid4())
            self.chain[self.id] = ""
            self.ids.append(self.id)

    def freq_dist(self, number=5):
        """
        Returns a NLTK FreqDist from a given list.
        :param number: Number of top most frequent items
        """
        @nldmethod
        def freq_dist_decorator(func):
            self._check_id(func)
            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def freq_dist_wrapper(_input=None):
                result = func(_input) if _input else func()
                if isinstance(result, list):
                    if self.logger:
                        self.logger.debug("Freq Dist : Getting frequencies...")
                    return FreqDist(result).most_common(number)
                else:
                    raise TypeError("The input to freq_dist must be of type list")
            return freq_dist_wrapper

        return freq_dist_decorator

    def named_entity(self, _func=None):
        """
        Applies the ne_chunk from NLTK
        """
        def named_entity_decorator(func):
            self._check_id(func)
            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def named_entity_wrapper(_input=None):
                result = func(_input) if _input else func()
                if isinstance(result, list) and isinstance(result[0], tuple):
                    return ne_chunk(result)

            return named_entity_wrapper
        if not _func:
            return named_entity_decorator
        else:
            return named_entity_decorator(_func)

    def pos_tagger(self, _func=None):
        """
        Takes a string or a list of strings and returns a NLTK pos_tag list.
        :param func:
        :return:
        """
        def pos_tagger_decorator(func):
            self._check_id(func)
            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def pos_wrapper(_input=None):
                result = func(_input) if _input else func()
                if isinstance(result, str):
                    if self.logger:
                        self.logger.info("POS Tagger : Input to pos tagger is of type string.")
                    return list(pos_tag(result.split()))
                elif isinstance(result, list):
                    if self.logger:
                        self.logger.info("POS Tagger : Input to pos tagger is of type list.")
                    return list(pos_tag(result))
                else:
                    raise TypeError("pos_tagger decorator only accepts string or list output, output received is %s" % type(result))
            return pos_wrapper
        if not _func:
            return pos_tagger_decorator
        else:
            return pos_tagger_decorator(_func)

    def n_grams(self, number):
        """
        Takes a string or list of strings and returns a list of ngrams.
        :param number: value N for the n-gram.
        :return:
        """
        @nldmethod
        def ngrams_decorator(func):
            self._check_id(func)
            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def ngrams_wrapper(_input=None):
                result = func(_input) if _input else func()
                if isinstance(result, str):
                    return list(ngrams(result.split(), number))
                elif isinstance(result, list):
                    return list(ngrams(result, number))
                else:
                    raise TypeError("n_grams decorator only accepts string or list output, output received is %s" % type(result))
            return ngrams_wrapper

        return ngrams_decorator

    def stem(self, _func=None):
        """
        Takes a list of tuples and applies the EnglishStemmer from NLTK stem.snowball.
        :param func:
        :return:
        """
        def stem_decorator(func):
            self._check_id(func)
            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def stem_wrapper(_input=None):
                stemmer = EnglishStemmer()
                result = func(_input) if _input else func()
                if isinstance(result, list):
                    if isinstance(result[0], tuple):
                        for i in range(len(result)):
                            result[i] = list(result[i])
                            result[i][0] = stemmer.stem(result[i][0])
                            result[i] = tuple(result[i])
                        return result
                    return [stemmer.stem(word) for word in result]
            return stem_wrapper
        if not _func:
            return stem_decorator
        else:
            return stem_decorator(_func)

    def lemmatize(self, _func=None):
        """
        Applies the WordNetLemmatizer from NLTK.
        return
        """

        def lemmatize_decorator(func):
            self._check_id(func)
            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def lemmatize_wrapper(_input=None):
                lemmatizer = WordNetLemmatizer()
                result = func(_input) if _input else func()
                if isinstance(result, list):
                    if len(result) > 0 and isinstance(result[0], tuple):
                        if self.logger:
                            self.logger.info('Lemmatize : input is tuple')
                        for i in range(len(result)):
                            result[i] = list(result[i])
                            result[i][0] = lemmatizer.lemmatize(result[i][0])
                            result[i] = tuple(result[i])
                        return result
                    return [lemmatizer.lemmatize(word) for word in result]
            return lemmatize_wrapper
        if not _func:
            return lemmatize_decorator
        else:
            return lemmatize_decorator(_func)

    def remove_stopwords(self, _func=None, *, punct=False, extra=[]):
        """
        Takes a list of strings and removes all strings in attribute self.stopwords. If punct True it also removes punctuation.
        The arguments punct and extra have to be specified when calling the function if they want to be set differently than default.
        :param punct: Whether or not to remove punctuation as well.
        :param extra: A list of strings to add extra stopwords to the ones already available, only for this run
        :return:
        """

        if extra:
            if not isinstance(extra, list) or  not isinstance(extra, tuple):
                raise TypeError("Extra stopwords have to be provided in a list or tuple.")

        def remove_stopwords_decorator(func):
            self._check_id(func)
            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def rm_stopwords_wrapper(_input=None):
                result = func(_input) if _input else func()
                if not isinstance(result, list):
                    raise TypeError("remove_stopwords decorator only accepts a list output, output received is %s" % type(result))
                if not punct:
                    return [word for word in result if word not in self.stopwords + list(extra)]
                else:
                    punctuation = set(string.punctuation)
                    return [word for word in result if word not in self.stopwords and word not in punctuation]

            return rm_stopwords_wrapper
        if not _func:
            return remove_stopwords_decorator
        else:
            return remove_stopwords_decorator(_func)

    def upper(self, _func=None):
        """
        Returns a string or list of strings as upper case.
        :param func:
        :return:
        """
        def upper_decorator(func):
            self._check_id(func)

            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def upper_wrapper(_input=None):
                result = func(_input) if _input else func()
                if isinstance(result, str):
                    return result.upper()
                elif isinstance(result, list):
                    return [word.upper() for word in result]
                else:
                    raise TypeError("upper decorator only accepts string or list output, output received is %s" % type(result))
            return upper_wrapper
        if not _func:
            return upper_decorator
        else:
            return upper_decorator(_func)

    def lower(self, _func=None):
        """
        Returns a string or list of strings as lower case.
        :param func:
        :return:
        """
        @nldmethod
        def lower_decorator(func):
            self._check_id(func)

            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def lower_wrapper(_input=None):
                result = func(_input) if _input else func()
                if isinstance(result, str):
                    return result.lower()
                elif isinstance(result, list):
                    return [word.lower() for word in result]
                else:
                    raise TypeError("lower decorator only accepts string or list output, output received is %s" % type(result))
            return lower_wrapper
        if not _func:
            return lower_decorator
        else:
            return lower_decorator(_func)

    def substitute(self, patterns):
        """
        Substitutes matching regex with a given string, can be applied on a string or a list.
        :param patterns: a tuple or list of tuples with the pattern at index 0 and new string at index 1.
        :return:
        """
        def sub_decorator(func):
            self._check_id(func)
            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def sub_wrapper(*args, **kwargs):
                import re
                result = func(*args, **kwargs)
                if self.logger:
                    self.logger.info("Substitue : patterns: %s", patterns)
                if isinstance(result, str):
                    for pattern in patterns:
                        old_word, new_word = pattern
                        result = re.sub(old_word, new_word, result)
                        return result
                elif isinstance(result, list):
                    for pattern in patterns:
                        old_word, new_word = pattern
                        result = [re.sub(old_word, new_word, word) for word in result]
                        return result
                else:
                    raise TypeError("substitute decorator only accepts string or list output, output received is %s" % type(result))
            return sub_wrapper

        return sub_decorator

    def apply_to_column(self, column_name):
        # TODO
        def apply_to_column_decorator(func):
            self._check_id(func)
            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def apply_to_column_wrapper(*args, **kwargs):
                raise NotImplementedError("This method is not developed / implemented yet.")

    def word_tokenizer(self, _func=None):
        """
        Applies NLTK word_tokenizer from tokenize
        :param func:
        :return:
        """
        def word_tokenizer_decorator(func):
            self._check_id(func)
            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def word_tokenizer_wrapper(_input=None):
                result = func(_input) if _input else func()
                if not isinstance(result, str):
                    raise TypeError("Decorator word_tokenizer only accepts string output, output received is %s" % type(result))
                try:
                    result = word_tokenize(result)
                    return result
                except LookupError:
                    raise LookupError("You miss the stopwords module from NLTK, which is required for NLD. Execute nltk.download('punkt') to download it.")
                return word_tokenize(result)
            return word_tokenizer_wrapper

        if not _func:
            return word_tokenizer_decorator
        else:
            return word_tokenizer_decorator(_func)

    def timeit(self, _func=None):
        """
        Times the execution of the previous decorators and appends to the list of run times.
        :param func: a function
        :return:
        """
        def timeit_decorator(func):
            self._check_id(func)
            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def timeit_wrapper(_input=None):
                t0 = time()
                result = func(_input) if _input else func()
                timing = time() - t0
                self.process_time = timing
                if self.logger:
                    self.logger.info("Timeit : Preprocessing took %.2f seconds", timing)
                if self.store_all_process_times:
                    self.all_process_times[func.__name__] = self.process_time
                return result
            return timeit_wrapper
        if not _func:
            return timeit_decorator
        else:
            return timeit_decorator(_func)

    def iterator(self, track_number=None):
        """
        Sets the iterable attribute if it was not set an returns the next item from it.
        This can be used to pass a list of sentences / texts to the next decorator for example.
        :param track_number: any string or number to keep track of iteration for different runs where the input function has the same name.
        :return:
        """
        def iterator_decorator(func):
            self._check_id(func)
            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def iterator_wrapper(_input=None):
                result = func(_input) if _input else func()
                if not isinstance(result, list):
                    raise TypeError("Decorator iterator_wrapper only accepts list output, output received is %s" % type(result))
                key_name = func.__name__ + str(track_number) if track_number else func.__name__
                if not self.iterable or key_name not in self.iterable:
                    self.iterable[key_name] = (item for item in result)
                try:
                    if self.logger:
                        self.logger.info("Iterable : key_name : %s", self.iterable[key_name])
                    return next(self.iterable[key_name])
                except StopIteration:
                    raise StopIteration("There are no more iterables")
            return iterator_wrapper
        return iterator_decorator

    def open_from_path(self, _func=None):
        """
        Opens a single file or all the files in a given directory.
        :param func:
        :return:
        """
        def open_from_path_decorator(func):
            self._check_id(func)
            self.chain[self.id] += func.__name__ + "-"

            @nldmethod
            def open_from_path_wrapper(_input=None):
                result = func(_input) if _input else func()
                try:
                    import os
                    if isinstance(result, str) and os.path.isfile(result):
                        with open(result) as output:
                            output = output.read()
                            return output
                    elif isinstance(result, str) and os.path.isdir(result):
                        output = [open(os.path.join(result, _file)).read() for _file in os.listdir(result)]
                        return output
                except:
                    raise
            return open_from_path_wrapper
        if not _func:
            return open_from_path_decorator
        else:
            return open_from_path_decorator(_func)

    def blank(self, func):
        """
        A blank decorator if you want to have also the last meaningful decorator in `self.chain`.
        :param func: a function
        :return:
        """
        self._check_id(func)
        self.chain[self.id] += func.__name__ + "-"

        @nldmethod
        def blank_wrapper(_input=None):
            result = func(_input) if _input else func()
            return result
        return blank_wrapper
