

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

training_data_file = "train_data.txt"
validation_data_file = "test_data_solution.txt"
test_data_file = "test_data.txt"


train_df = pd.read_csv(training_data_file, delimiter=" ::: ", names=["index", "movie_name", "genre", "description"])
validation_df = pd.read_csv(validation_data_file, delimiter=" ::: ", names=["index", "movie_name", "genre", "description"])
test_df = pd.read_csv(test_data_file, delimiter=" ::: ", names=["index", "movie_name", "description"])

combined_df = pd.concat([train_df, validation_df])

tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf_vectorizer.fit_transform(combined_df['description'])
y = combined_df['genre']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = SVC(kernel='linear')

classifier.fit(X_train, y_train)
X_test = tfidf_vectorizer.transform(test_df['description'])
y_pred = classifier.predict(X_test)


test_df['predicted_genre'] = y_pred

while True:
    movie_name = input("Enter a movie name (or 'quit' to exit): ")
    if movie_name.lower() == 'quit':
        break
    else:
        movie = test_df[test_df['movie_name'] == movie_name]
        if not movie.empty:
            predicted_genre = movie.iloc[0]['predicted_genre']
            print(f"Predicted Genre for '{movie_name}': {predicted_genre}")
        else:
            print(f"Movie '{movie_name}' not found in the test dataset.")

print("\nTest Results:")
print(classification_report(validation_df['genre'], classifier.predict(X_val)))


# In[ ]:


