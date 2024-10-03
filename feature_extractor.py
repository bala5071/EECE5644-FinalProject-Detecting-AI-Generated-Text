import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
from fast_ml.model_development import train_valid_test_split
import pickle

# Downloading stop words and wordnet
nltk.download("wordnet")
nltk.download("stopwords")

# Making a stopwords list
cachedStopWords = stopwords.words("english")
cachedStopWords += [i for i in string.punctuation]  # Adding punctuations to stop words
cachedStopWords += ["``"]


# To remove punctuations
def clean_word(word):
    puncts = [i for i in string.punctuation]
    for i in word:
        if i in puncts:
            word = word.replace(i, "")
    return word


# To check if the word is a number, stop word or has length less than or equal to 2
def check_good_word(word):
    if any(i.isdigit() for i in word):
        return False
    return word not in cachedStopWords and len(word) > 2


# For text preprocessing
def textPreprocessingTokenizer(text):
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    stemmer = nltk.stem.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    text = tokenizer.tokenize(text.strip())

    final_text = []
    for word in text:
        word = clean_word(word)
        if check_good_word(word):
            final_text.append(word)
    final_text = " ".join([word for word in final_text])
    final_text = " ".join(
        lemmatizer.lemmatize(stemmer.stem(token))
        for token in tokenizer.tokenize(final_text)
        if len(token) > 2 and token not in cachedStopWords
    ).split(" ")

    return final_text


if __name__ == "__main__":
    # Importing the dataset
    df = pd.read_csv("./mainDataset.csv")
    df = df.sample(frac=1)  # Shuffling the dataset
    df = df.fillna(np.nan)
    df.dropna(
        subset=["abstract"], inplace=True
    )  # Removing rows containing empty values

    # Initializing TFIDF Vectorizer
    tfidf = TfidfVectorizer(
        tokenizer=textPreprocessingTokenizer,
        ngram_range=(1, 1),
        stop_words="english",
        max_features=10000,
    )

    tfidf = tfidf.fit(df["abstract"])
    tfidf_df = tfidf.transform(df["abstract"])

    # Converting TFIDF to csv
    tfidf_df = pd.DataFrame(
        tfidf_df.todense().round(1), columns=tfidf.get_feature_names_out()
    )
    tfidf_df["y_vector"] = df["label"]
    tfidf_df.to_csv("Features/TFIDF/TFIDF.csv", index=False)

    # Initializing TF Vectorizer
    tf = CountVectorizer(
        tokenizer=textPreprocessingTokenizer,
        ngram_range=(1, 1),
        stop_words="english",
        max_features=10000,
    )

    tf = tf.fit(df["abstract"])
    tf_df = tf.transform(df["abstract"])

    # Converting TF to csv
    tf_df = pd.DataFrame(tf_df.todense().round(1), columns=tf.get_feature_names_out())
    tf_df["y_vector"] = df["label"]
    tf_df.to_csv("Features/TF/TF.csv", index=False)

    # Splitting TFIDF to train, test and validation sets
    x_train, y_train, x_valid, y_valid, x_test, y_test = train_valid_test_split(
        tfidf_df, target="y_vector", train_size=0.8, valid_size=0.1, test_size=0.1
    )

    # Exporting TFIDF train, test and validation sets
    x_train.to_csv("Features/TFIDF/x_train.csv", index=False)
    y_train.to_csv("Features/TFIDF/y_train.csv", index=False)
    x_valid.to_csv("Features/TFIDF/x_valid.csv", index=False)
    y_valid.to_csv("Features/TFIDF/y_valid.csv", index=False)
    x_test.to_csv("Features/TFIDF/x_test.csv", index=False)
    y_test.to_csv("Features/TFIDF/y_test.csv", index=False)

    # Splitting TF to train, test and validation sets
    x2_train, y2_train, x2_valid, y2_valid, x2_test, y2_test = train_valid_test_split(
        tf_df, target="y_vector", train_size=0.8, valid_size=0.1, test_size=0.1
    )

    # Exporting TF train, test and validation sets
    x2_train.to_csv("Features/TF/x_train.csv", index=False)
    y2_train.to_csv("Features/TF/y_train.csv", index=False)
    x2_valid.to_csv("Features/TF/x_valid.csv", index=False)
    y2_valid.to_csv("Features/TF/y_valid.csv", index=False)
    x2_test.to_csv("Features/TF/x_test.csv", index=False)
    y2_test.to_csv("Features/TF/y_test.csv", index=False)

    # Printing shape of TFIDF sets to verify dimensions
    print(f"\nTFIDF: {tfidf_df.shape}")
    print(f"TFIDF_x_train: {x_train.shape}")
    print(f"TFIDF_y_train: {y_train.shape}")
    print(f"TFIDF_x_valid: {x_valid.shape}")
    print(f"TFIDF_y_valid: {y_valid.shape}")
    print(f"TFIDF_x_test: {x_test.shape}")
    print(f"TFIDF_y_test: {y_test.shape}")

    # Printing shape of TF sets to verify dimensions
    print(f"\nTF: {tf_df.shape}")
    print(f"TF_x_train: {x2_train.shape}")
    print(f"TF_y_train: {y2_train.shape}")
    print(f"TF_x_valid: {x2_valid.shape}")
    print(f"TF_y_valid: {y2_valid.shape}")
    print(f"TF_x_test: {x2_test.shape}")
    print(f"TF_y_test: {y2_test.shape}")

    # Saving the vectorizers
    with open("vectorizers/tfidf.pkl", "wb") as f:
        vec_tfidf = pickle.dump(tfidf, f)

    with open("vectorizers/tf.pkl", "wb") as f:
        vec_tf = pickle.dump(tf, f)
