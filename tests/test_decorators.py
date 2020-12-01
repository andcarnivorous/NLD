import unittest
from unittest import TestCase
from nld.nld import NLD

text = open("loremipsum.txt").read()

def return_text(text):
    return text

def return_none():
    return None

class TestDecorators(TestCase):

    def setUp(self):
        self.nldecorator = NLD()

    def test_word_tokenizer(self):
        func = self.nldecorator.word_tokenizer(return_text)
        result = func(text)
        self.assertEqual(len(result), 571)
        self.assertTrue(isinstance(result, list))

    def test_word_tokenizer_exception(self):
        func = self.nldecorator.word_tokenizer(return_none)
        with self.assertRaises(TypeError):
            result = func(text)

    def test_lower(self):
        func = self.nldecorator.lower(return_text)
        result = func(text)
        self.assertTrue(result.islower())

    def test_lower_list(self):

        @self.nldecorator.lower
        @self.nldecorator.word_tokenizer
        def return_text(text):
            return text

        result = return_text(text[:8])
        self.assertTrue(all(x.islower() for x in result))

    def test_upper(self):
        func = self.nldecorator.upper(return_text)
        result = func(text)
        self.assertTrue(result.isupper())

    def test_upper_list(self):

        @self.nldecorator.upper
        @self.nldecorator.word_tokenizer
        def return_text(text):
            return text

        result = return_text(text[8])
        self.assertTrue(all(x.isupper() for x in result))