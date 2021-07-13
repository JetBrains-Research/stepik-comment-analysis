import re
import time
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
from nltk.corpus import stopwords

STOPWORDS = stopwords.words("russian")
string.punctuation += "—"
string.punctuation += "№"
string.punctuation = string.punctuation.replace("?", "")

PATTERN = ["?", "проблема", "почему"]


class Cleaner:
    def __init__(self, data):
        self.text = data

    def remove_tags(self):
        tags = re.compile(r"<.*?>")
        self.text = tags.sub(" ", self.text)

    def remove_users(self):
        users = re.compile(r"@[\w_]+")
        self.text = users.sub("", self.text)

    def remove_links(self):
        links = re.compile(r"https?://\S+")
        self.text = links.sub("", self.text)

    def remove_expletives(self):
        expletives = re.compile(r"[А-яё]+@\w+")
        self.text = expletives.sub("", self.text)

    def remove_whitespace(self):
        self.text = " ".join(self.text.split())

    def remove_emojis(self):
        emojis = re.compile(
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
        self.text = re.sub(emojis, "", self.text)

    def remove_symbols(self):
        punctuation = re.compile("[%s]" % re.escape(string.digits + string.punctuation))
        text_lower = self.text.lower()
        text_lower = punctuation.sub(" ", text_lower)
        text_lower = re.sub(r"[a-z]+", "", text_lower)
        self.text = re.sub(r"\s+", " ", text_lower)


class LemmatizerNatasha:
    def __init__(self, data):
        self.text = data
        self.stop_words = STOPWORDS
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)

    def lemmatizer(self):
        doc = Doc(self.text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)

        return [_.lemma for _ in doc.tokens if _.lemma not in self.stop_words]


def top_level_comments(df):
    return df[df.parent_id.isna()][["comment_id", "step_id", "text"]]


def clean_texts(texts):
    start_time = time.time()
    cleaned_texts = np.array([])
    for text in texts:
        cleaned_text = Cleaner(text)
        cleaned_text.remove_tags()
        cleaned_text.remove_expletives()
        cleaned_text.remove_links()
        cleaned_text.remove_users()
        cleaned_text.remove_whitespace()
        cleaned_text.remove_emojis()
        cleaned_text.remove_symbols()
        cleaned_texts = np.append(cleaned_texts, cleaned_text.text)
    print("--- clean_texts: %s seconds ---" % (time.time() - start_time))
    return cleaned_texts


def lemmatize_texts(texts):
    start_time = time.time()
    lemmatized_texts = []
    for text in texts:
        lemmatized_text = LemmatizerNatasha(text)
        lemmatized_texts.append(lemmatized_text.lemmatizer())
    print("--- lemmatize_texts: %s seconds ---" % (time.time() - start_time))
    return lemmatized_texts


def identity_tokenizer(tokens):
    return tokens


def vectorize_texts(texts):
    start_time = time.time()
    vect = TfidfVectorizer(tokenizer=identity_tokenizer, min_df=2, lowercase=False)
    vectorized_texts = vect.fit_transform(texts)
    print("--- vectorized_texts: %s seconds ---" % (time.time() - start_time))
    return vectorized_texts.toarray()


def is_question(texts):
    questions = []
    questions_idx = []
    for idx, text in enumerate(texts):
        if any(word in PATTERN for word in text):
            text = [word for word in text if "?" not in word]
            if text:
                questions.append(text)
                questions_idx.append(idx)
    questions_dict = {k: v for k, v in enumerate(questions_idx)}
    return questions_dict, questions


def get_similar_questions(vectors, treshold=0.75):
    similarities = cosine_similarity(vectors)
    similarities = np.triu(similarities, k=1)
    similar_vectors = np.argwhere(similarities > treshold)
    return similar_vectors
