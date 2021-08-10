import re
import numpy as np
from preprocessing import Cleaner, LemmatizerNatasha


def is_question(df):
    comments = df["text"].values
    if "use_problem" in df:
        use_problem = df["use_problem"].values
    else:
        use_problem = np.full(df.shape[0], True)

    cleaner = Cleaner(comments, method="bert")
    cleaned_text = cleaner.get_cleaned_texts()

    mask = Comments(comments, use_problem).question()
    mask_noun, lemmas = LemmatizerNatasha(cleaned_text).get_questions_lemmas()

    df["cleaned_text"] = cleaned_text
    df["lemmas"] = lemmas
    df["is_question"] = mask
    return df[mask_noun].sort_values("is_question", ascending=False).reset_index(drop=True)


def top_level_comments(df):
    return df[df["parent_id"].isna()][["comment_id", "step_id", "text"]]


class Comments:
    def __init__(self, comments, use_problem):
        self.comments = comments
        self.words_without_problem = r"\?|почему|не получается|не знаю|не могу|ошибк|помогите|подскажи|подсказать"
        self.words_with_problem = self.words_without_problem + r"проблем"
        self.use_problem = use_problem

    def search_words(self, comment, problem_ind):
        if problem_ind:
            pattern = re.compile(self.words_with_problem)
        else:
            pattern = re.compile(self.words_without_problem)
        return bool(pattern.search(comment.lower()))

    def drop_answer(self, comment):
        comment = comment.lower()
        if re.search(r"помогло", comment):
            not_answer = bool(re.search(r" не помогло", comment))
        else:
            not_answer = True
        return not_answer

    def question(self):
        mask = []
        for comment, problem_ind in zip(self.comments, self.use_problem):
            if self.search_words(comment, problem_ind) & self.drop_answer(comment):
                mask.append(True)
            else:
                mask.append(False)
        return mask
