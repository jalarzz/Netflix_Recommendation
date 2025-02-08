# Netflix Movie Recommender System

## Overview
This project is a **Netflix Movie Recommendation System** that suggests similar movies based on a given title using textual content such as **description**, **director**, **cast**, **listed_in (genre)**, and **type** (movie/show type). It employs **TF-IDF** (Term Frequency-Inverse Document Frequency) vectorization and **Cosine Similarity** to find the most similar movies in the dataset.

## Features
- **Data Preprocessing**: Cleans the text data by removing special characters and stopwords.
- **TF-IDF Vectorization**: Converts text data into numerical format for machine learning.
- **Cosine Similarity**: Measures the similarity between movies based on their textual features.
- **Flexible Recommendations**: Allows specifying the number of recommendations and adjusts the recommendation by considering different features like genre, description, director, etc.
- **Exploratory Analysis**: Identifies directors with the most movies, analyzes the distribution of movie durations, and provides other interesting insights about the dataset.

## How it Works
1. **Text Preprocessing**: The text is cleaned by removing non-alphabetic characters and common stopwords to focus on meaningful words.
2. **Feature Combination**: Descriptions, directors, cast, genres, and type fields are combined into a single feature column for each movie.
3. **TF-IDF Matrix**: The combined features are transformed into numerical data using TF-IDF, capturing the importance of each word across the dataset.
4. **Cosine Similarity**: A similarity matrix is computed based on the TF-IDF matrix to measure how similar movies are to each other.
5. **Recommendation Engine**: Given a movie title, the system calculates similarity scores and recommends the most similar movies.

## Dataset
This recommender system requires a dataset containing information about movies, including:
- `title`
- `description`
- `director`
- `cast`
- `listed_in` (genre)
- `type` (e.g., movie, show)

