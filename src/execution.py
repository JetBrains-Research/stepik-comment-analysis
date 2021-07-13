import load_data
import preprocessing

# add step_id value
STEP_ID = 6532

df_comm = load_data.load_comments()
df_comm = preprocessing.top_level_comments(df_comm)
df = df_comm[df_comm.step_id == STEP_ID]

texts = df.text.values
cleaned_texts = preprocessing.clean_texts(texts)
lemmatized_texts = preprocessing.lemmatize_texts(cleaned_texts)
questions_dict, questions = preprocessing.is_question(lemmatized_texts)
print(len(questions_dict))
vectorized_questions = preprocessing.vectorize_texts(questions)
similar_questions = preprocessing.get_similar_questions(vectorized_questions)

for pair in similar_questions:
    pair_indexes = [questions_dict[k] for k in pair]
    print("")
    print(cleaned_texts[pair_indexes])
