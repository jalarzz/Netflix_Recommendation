import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
from nltk.corpus import stopwords

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import nltk
nltk.download('stopwords')
class NetflixRecommender:
    def __init__(self, df):
        self.df = df.copy()
        self.df['description'] = self.df['description'].fillna('')
        self.df['director'] = self.df['director'].fillna('')
        self.df['cast'] = self.df['cast'].fillna('')
        self.df['listed_in'] = self.df['listed_in'].fillna('')

        # Create a combined features column
        self.df['combined_features'] = self.create_combined_features()

        #vectorizing
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_features'])


        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def clean_text(self, text):
        """Cleaning data"""

        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [w for w in words if w not in stop_words]

        return ' '.join(words)

    def create_combined_features(self):
        """combining different features into a single string"""
        return (self.df['description'].apply(self.clean_text) + ' ' +
                self.df['director'].apply(self.clean_text) + ' ' +
                self.df['cast'].apply(self.clean_text) + ' ' +
                self.df['listed_in'].apply(self.clean_text) + ' ' +
                self.df['type'].apply(self.clean_text))

    def get_recommendations(self, title, n_recommendations=5):
        """Get n_recommendations similar titles based on the input title"""
        try:

            idx = self.df[self.df['title'] == title].index[0]
            similarity_scores = list(enumerate(self.similarity_matrix[idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            similar_titles = similarity_scores[1:n_recommendations+1]

            recommendations = []
            for i, score in similar_titles:
                recommendations.append({
                    'title': self.df.iloc[i]['title'],
                    'type': self.df.iloc[i]['type'],
                    'description': self.df.iloc[i]['description'],
                    'similarity_score': score,
                    'genre': self.df.iloc[i]['listed_in']
                })

            return recommendations

        except IndexError:
            return f"Title '{title}' not found in the dataset."
