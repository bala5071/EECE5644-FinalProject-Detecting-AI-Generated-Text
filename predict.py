import pickle
from feature_extractor import textPreprocessingTokenizer
from sklearn.preprocessing import StandardScaler
import pandas as pd

FEATURE = "TFIDF"  # Select the feautres (TF/TFIDF)
TEXT = """"""  # Input Text

# Load Vectorizer
with open(
    f"vectorizers/{FEATURE}.pkl",
    "rb",
) as f:
    vectorizer = pickle.load(f)

features = vectorizer.transform([TEXT])  # Generating Features
features = pd.DataFrame(features.todense(), columns=vectorizer.get_feature_names_out())

# Load Normalizer
with open(
    f"vectorizers/sc.pkl",
    "rb",
) as f:
    sc = pickle.load(f)
features = sc.transform(features)

# Load Model
with open(
    f"models/Random Forest/{FEATURE}/Random_Forest_{FEATURE}_{75}.pkl",
    "rb",
) as f:
    classifier = pickle.load(f)

# Test Model
y_pred = classifier.predict(features)
if y_pred == 1:
    print("Computer Generated")
else:
    print("Human Written")
