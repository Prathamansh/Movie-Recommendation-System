import pandas as pd
import numpy as np
import re
import ast
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
movies = pd.read_csv(r"C:\Users\PRATHAMANSH JASROTIA\OneDrive\Desktop\all unneccassary works here\ml-movie-rec-project\Dataset\tmdb_5000_movies.csv")
credits = pd.read_csv(r"C:\Users\PRATHAMANSH JASROTIA\OneDrive\Desktop\all unneccassary works here\ml-movie-rec-project\Dataset\tmdb_5000_credits.csv")

movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# JSON parsing functions
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

def fetch_director(text):
    return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']

def collapse(L):
    return [i.replace(" ", "") for i in L]

# Apply parsing
movies['genres'] = movies['genres'].apply(convert).apply(collapse)
movies['keywords'] = movies['keywords'].apply(convert).apply(collapse)
movies['cast'] = movies['cast'].apply(convert).apply(lambda x: x[:3]).apply(collapse)
movies['crew'] = movies['crew'].apply(fetch_director).apply(collapse)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Create tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies[['movie_id', 'title', 'tags']].copy()  # ✅ avoid SettingWithCopyWarning
new['tags'] = new['tags'].apply(lambda x: " ".join(x))

# Preprocess text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

new['tags'] = new['tags'].apply(preprocess_text)
new = new[new['tags'].apply(lambda x: len(x.strip()) > 0)]

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english', max_df=0.8, min_df=2, ngram_range=(1, 2))
vector = cv.fit_transform(new['tags']).toarray()
similarity = cosine_similarity(vector)

# Recommendation with flexible search
def recommend(movie):
    matches = new[new['title'].str.lower().str.contains(movie.lower())]

    if matches.empty:
        print(f"\n '{movie}' not found in dataset.")
        return

    print("\n Did you mean one of these?")
    for i, title in enumerate(matches['title'].values, start=1):
        print(f"{i}. {title}")

    try:
        selected_index = int(input("\nEnter the number of the correct title: ")) - 1
        if selected_index < 0 or selected_index >= len(matches):
            print(" Invalid selection.")
            return
    except ValueError:
        print(" Please enter a valid number.")
        return

    selected_title = matches.iloc[selected_index]['title']
    index = new[new['title'] == selected_title].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    print(f"\n Top 5 movies similar to **{selected_title}**:")
    for i in distances[1:6]:
        print(f"→ {new.iloc[i[0]].title}")

# Run the recommender
user_input = input(" Enter a movie title: ")
recommend(user_input)
