import streamlit as st
import pandas as pd
import numpy as np
import requests

# File URLs on GitHub
TOP_100_MOVIES_URL = "https://raw.githubusercontent.com/YichengLiPaul/CS598-Proj4/main/top_100_movies.txt"
SIMILARITY_MATRIX_URL = "https://raw.githubusercontent.com/YichengLiPaul/CS598-Proj4/main/top_30_similarity_matrix.csv"

# Load the top 100 movies and similarity matrix
@st.cache_data
def load_data():
    # Load top 100 movies (MovieID list)
    top_100_movies = pd.read_csv(TOP_100_MOVIES_URL, header=None, names=["MovieID"])
    top_100_movies = top_100_movies["MovieID"].tolist()

    # Load similarity matrix
    similarity_matrix = pd.read_csv(SIMILARITY_MATRIX_URL, index_col=0)
    return top_100_movies, similarity_matrix

top_100_movies, similarity_matrix = load_data()

# Function to generate poster URL
def get_poster_url(movie_id):
    base_url = "https://liangfgithub.github.io/MovieImages/"
    poster_url = f"{base_url}{movie_id}.jpg?raw=true"
    return poster_url

# myIBCF function to recommend top 10 movies
def myIBCF(new_user, similarity_matrix):
    predictions = pd.Series(index=similarity_matrix.columns, dtype=float)
    for movie in similarity_matrix.columns:
        if pd.isna(new_user[movie]):  # Only predict for unrated movies
            similar_movies = similarity_matrix.loc[movie].dropna()
            rated_movies = similar_movies.index.intersection(new_user.dropna().index)
            if len(rated_movies) > 0:
                weights = similar_movies[rated_movies]
                ratings = new_user[rated_movies]
                numerator = (weights * ratings).sum()
                denominator = weights.sum()
                if denominator > 0:
                    predictions[movie] = numerator / denominator
    return predictions.dropna().sort_values(ascending=False).head(10)

# Streamlit UI
st.title("Movie Recommendation System")
st.write("Rate movies below, and get personalized recommendations!")

# Step 1: Display the top 100 movies for rating
st.subheader("Please rate the following movies:")
user_ratings = pd.Series(index=top_100_movies, dtype=float)

# Use columns to display movies in a grid
cols = st.columns(5)  # 5 movies per row
for idx, movie_id in enumerate(top_100_movies):
    with cols[idx % 5]:
        st.image(get_poster_url(movie_id), caption=f"Movie ID: {movie_id}", width=120)
        rating = st.slider(f"Rate {movie_id}", 1, 5, value=0, step=1, key=f"rating_{movie_id}")
        if rating > 0:
            user_ratings[movie_id] = rating

# Submit button
if st.button("Get Recommendations"):
    st.subheader("Your Recommendations:")
    # Run the IBCF function
    recommendations = myIBCF(user_ratings, similarity_matrix)
    if recommendations.empty:
        st.write("No recommendations found. Please rate more movies!")
    else:
        for idx, movie_id in enumerate(recommendations.index):
            st.image(get_poster_url(movie_id), caption=f"Movie ID: {movie_id}", width=120)
            st.write(f"#{idx + 1}: Movie ID {movie_id} (Predicted rating: {recommendations[movie_id]:.2f})")
