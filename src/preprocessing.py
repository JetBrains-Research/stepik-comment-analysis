import re
import string
from functools import reduce
from typing import List, Callable

import numpy as np
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

STOPWORDS = stopwords.words("russian")
STOPWORDS_NOUN = ["пож", "решение", "подскажите", "проблема", "код", "программа", "ошибка", "что"]

ENG = ["failed", "test", "error", "wrong", "фэйл", "fail"]
string.punctuation += "—"
string.punctuation += "№"
symbols = re.sub("[!,.?]", "", string.punctuation)


class BertEmbedding:
    def __init__(self, df):
        self.df = df
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    def evaluate(self):
        if "cleaned_text" in self.df:
            cleaned_text = self.df.cleaned_text.values
        else:
            cleaner = Cleaner(self.df.text.values, method="bert")
            cleaned_text = cleaner.get_cleaned_texts()
        embeddings = self.model.encode(cleaned_text)
        return embeddings


def identity_tokenizer(tokens):
    return tokens


class TFIDFEmbedding:
    def __init__(self, df):
        self.df = df
        self.vect = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)

    def evaluate(self):
        if "lemmas" in self.df:
            cleaner = Cleaner(self.df.lemmas.values, method="tfidf")
            lemmatized_texts = cleaner.get_cleaned_texts()
        else:
            cleaner = Cleaner(self.df.text.values, method="tfidf")
            cleaned_text = cleaner.get_cleaned_texts()
            lematizer = LemmatizerNatasha(cleaned_text)
            lemmatized_texts = lematizer.get_lemmas()
        vectorized_texts = self.vect.fit_transform(lemmatized_texts).toarray()
        return vectorized_texts


class Cleaner:
    def __init__(self, texts, method):
        self.texts = texts
        self.method = method
        self.users = re.compile(r"@[\w_]+")
        self.expletives = re.compile(r"[А-яё]+@\w+")
        self.punctuation = re.compile(r"[%s]" % re.escape(string.punctuation))
        self.symbols = re.compile(r"[%s]" % re.escape(symbols))
        self.english = re.compile(r"[A-z]")
        self.digits = re.compile(r"[%s]" % re.escape(string.digits))
        self.emojis = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
            "]+",
            re.UNICODE,
        )

    def remove_users(self, text):
        return self.users.sub("", text)

    def remove_expletives(self, text):
        return self.expletives.sub("", text)

    @staticmethod
    def remove_whitespace(text):
        text = text.replace(u"\ufeff", "")
        return " ".join(text.split())

    def remove_emojis(self, text):
        return self.emojis.sub("", text)

    def remove_english(self, text):
        return self.english.sub("", text)

    def remove_digits(self, text):
        return self.digits.sub("", text)

    def remove_punctuation(self, text):
        text = self.punctuation.sub(" ", text.lower())
        text = re.sub(r"\s+", " ", text)
        return text

    def remove_symbols(self, text):
        text = self.symbols.sub(" ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def get_base_cleaners(self) -> List[Callable]:
        return [
            self.remove_expletives,
            self.remove_users,
            self.remove_emojis,
            self.remove_symbols,
            self.remove_whitespace,
        ]

    def get_tfidf_cleaners(self) -> List[Callable]:
        return [
            self.remove_expletives,
            self.remove_users,
            self.remove_whitespace,
            self.remove_emojis,
            self.remove_english,
            self.remove_digits,
            self.remove_punctuation,
        ]

    def get_cleaned_texts(self):
        cleaned_text = []
        cleaners = self.get_tfidf_cleaners() if self.method == "tfidf" else self.get_base_cleaners()
        cleaned_text += [reduce(lambda t, c: c(t), cleaners, text).strip() for text in self.texts]
        return cleaned_text


class LemmatizerNatasha:
    def __init__(self, texts):
        self.texts = texts
        self.stopwords = STOPWORDS
        self.stopwords_noun = STOPWORDS_NOUN
        self.eng = ENG
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)

    def lemmatize(self, text):
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        return np.array([(_.lemma, _.pos) for _ in doc.tokens if _.lemma not in self.stopwords])

    def lemmatize_texts(self):
        lemmas_pos = []
        for text in self.texts:
            lemmatized_text = self.lemmatize(text)
            if not lemmatized_text.size:
                lemmatized_text = np.array([["", ""]])
            lemmas_pos.append(lemmatized_text)
        return lemmas_pos

    def get_questions_lemmas(self):
        lemmas_pos = self.lemmatize_texts()
        noun_texts = []
        lemmas_texts = []
        for i in lemmas_pos:
            words = i[:, 0]
            pos = i[:, 1]
            text = " ".join(words)
            noun_text = words[
                (any([substr in text for substr in self.eng]) | (pos == "NOUN"))
                & (~np.isin(words, self.stopwords_noun))
                & (np.char.str_len(words) > 1)
            ]
            noun_texts.append(noun_text.size > 0)
            lemmas_texts.append(text)
        return noun_texts, lemmas_texts

    def get_lemmas(self):
        lemmas_pos = self.lemmatize_texts()
        lemmas_texts = []
        for i in lemmas_pos:
            words = i[:, 0]
            text = " ".join(words)
            lemmas_texts.append(text)
        return lemmas_texts
