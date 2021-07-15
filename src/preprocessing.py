import re
import time
import string
from bs4 import BeautifulSoup
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

STOPWORDS = stopwords.words("russian")
string.punctuation += "—"
string.punctuation += "№"


class Cleaner:
    def __init__(self, texts):
        self.texts = texts
        self.users = re.compile(r"@[\w_]+")
        self.links = re.compile(r"https?://\S+")
        self.expletives = re.compile(r"[А-яё]+@\w+")
        self.symbols = re.compile(
            "[%sa-z]" % re.escape(string.digits + string.punctuation)
        )
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

    @staticmethod
    def remove_tags(text):
        return BeautifulSoup(text, "lxml").text

    def remove_users(self, text):
        return self.users.sub("", text)

    def remove_links(self, text):
        return self.links.sub("", text)

    def remove_expletives(self, text):
        return self.expletives.sub("", text)

    @staticmethod
    def remove_whitespace(text):
        return " ".join(text.split())

    def remove_emojis(self, text):
        return self.emojis.sub("", text)

    def remove_symbols(self, text):
        text_lower = text.lower()
        text_lower = self.symbols.sub("", text_lower)
        text_lower = re.sub(r"\s+", " ", text_lower)
        return text_lower

    def clean_texts(self):
        start_time = time.time()
        cleaned_texts = []
        for text in self.texts:
            cleaned_text = self.remove_tags(text)
            cleaned_text = self.remove_expletives(cleaned_text)
            cleaned_text = self.remove_links(cleaned_text)
            cleaned_text = self.remove_users(cleaned_text)
            cleaned_text = self.remove_whitespace(cleaned_text)
            cleaned_text = self.remove_emojis(cleaned_text)
            cleaned_text = self.remove_symbols(cleaned_text)
            cleaned_texts.append(cleaned_text)
        print("--- clean_texts: %s seconds ---" % (time.time() - start_time))
        return cleaned_texts


class LemmatizerNatasha:
    def __init__(self, texts):
        self.texts = texts
        self.stop_words = STOPWORDS
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
        return [_.lemma for _ in doc.tokens if _.lemma not in self.stop_words]

    def lemmatize_texts(self):
        start_time = time.time()
        lemmatized_texts = []
        for text in self.texts:
            lemmatized_text = self.lemmatize(text)
            lemmatized_texts.append(lemmatized_text)
        print("--- lemmatize_texts: %s seconds ---" % (time.time() - start_time))
        return lemmatized_texts


def identity_tokenizer(tokens):
    return tokens


def vectorize_texts(texts):
    cleaner = Cleaner(texts)
    cleaned_text = cleaner.clean_texts()
    lematizer = LemmatizerNatasha(cleaned_text)
    lemmatized_texts = lematizer.lemmatize_texts()

    vect = TfidfVectorizer(tokenizer=identity_tokenizer, min_df=2, lowercase=False)
    vectorized_texts = vect.fit_transform(lemmatized_texts)
    return vectorized_texts.toarray()
