import pandas as pd
import re
import ast
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class MovieRecommender:
    def __init__(self, movies_path, credits_path):
        movies = pd.read_csv(movies_path)
        credits = pd.read_csv(credits_path)

        self.movies = movies.copy()
        self.credits = credits.copy()

        self.movies['norm_title'] = self.movies['title'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
        self.credits['norm_title'] = self.credits['title'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

        self.movies = self.movies.merge(self.credits[['norm_title', 'cast', 'crew']], on='norm_title', how='left')

        self.title_lookup = {self._normalize_title(t): t for t in self.movies['title']}
        self._preprocess()
        self._vectorize()

    def _normalize_title(self, title):
        return re.sub(r'[^a-z0-9]', '', str(title).lower())

    def _preprocess(self):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\w+')

        def tokenize(text):
            return ' '.join([lemmatizer.lemmatize(w) for w in tokenizer.tokenize(str(text).lower()) if w not in stop_words])

        self.movies['overview'] = self.movies['overview'].fillna('').apply(tokenize)

        def parse_keywords(obj):
            try:
                keywords = [k['name'] for k in ast.literal_eval(obj)]
                return ' '.join([lemmatizer.lemmatize(k.replace(' ', '').lower()) for k in keywords if k.lower() not in stop_words])
            except:
                return ''
        self.movies['keywords'] = self.movies['keywords'].fillna('').apply(parse_keywords)

        def parse_cast(obj):
            try:
                cast = [c['name'] for c in ast.literal_eval(obj)][:5]
                return ' '.join([lemmatizer.lemmatize(c.replace(' ', '').lower()) for c in cast])
            except:
                return ''
        self.movies['cast'] = self.movies['cast'].fillna('').apply(parse_cast)

        def parse_director(obj):
            try:
                crew = ast.literal_eval(obj)
                for member in crew:
                    if member.get('job') == 'Director':
                        return lemmatizer.lemmatize(member['name'].replace(' ', '').lower())
            except:
                return ''
        self.movies['crew'] = self.movies['crew'].fillna('').apply(parse_director)

        self.movies['tags'] = (
            self.movies['overview'] + ' ' +
            self.movies['genres'].fillna('') + ' ' +
            self.movies['keywords'] + ' ' +
            self.movies['cast'] + ' ' +
            self.movies['crew']
        )

    def _vectorize(self):
        vect = TfidfVectorizer(max_features=5000)
        tfidf = vect.fit_transform(self.movies['tags']).toarray()
        self.similarity = cosine_similarity(tfidf)

    def recommend(self, title, top_n=5):
        norm = self._normalize_title(title)
        if norm not in self.title_lookup:
            close = get_close_matches(norm, self.title_lookup.keys(), n=1, cutoff=0.6)
            if not close:
                return []
            norm = close[0]

        matched = self.title_lookup[norm]
        idx = self.movies[self.movies['title'] == matched].index[0]
        sims = sorted(list(enumerate(self.similarity[idx])), key=lambda x: -x[1])[1:top_n+1]

        return [
            (
                self.movies.iloc[i]['title'],
                round(score, 2),
                self.movies.iloc[i].get('poster_url', '')
            )
            for i, score in sims
        ]
