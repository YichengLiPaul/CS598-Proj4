import streamlit as st
import pandas as pd
import numpy as np

TOP_100_MOVIES_URL = "https://raw.githubusercontent.com/YichengLiPaul/CS598-Proj4/main/top_100_movies.txt"
SIMILARITY_MATRIX_URL = "https://raw.githubusercontent.com/YichengLiPaul/CS598-Proj4/main/top_30_similarity_matrix.csv"
MOVIES_METADATA_URL = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

@st.cache_data
def load_data():
    # Load top 100 movies (MovieID list)
    top_100_movies = pd.read_csv(TOP_100_MOVIES_URL, header=None, names=["MovieID"])
    top_100_movies["MovieID"] = top_100_movies["MovieID"].str.replace("m", "", regex=False).astype(int)

    # Load similarity matrix
    similarity_matrix = pd.read_csv(SIMILARITY_MATRIX_URL, index_col=0)
    similarity_matrix.columns = similarity_matrix.columns.str.replace("m", "", regex=False).astype(int)
    similarity_matrix.index = similarity_matrix.index.str.replace("m", "", regex=False).astype(int)
    
    # Load movies.dat for metadata
    movies_metadata = pd.read_csv(
        MOVIES_METADATA_URL, sep="::", header=None, engine="python", 
        names=["MovieID", "Title", "Genres"]
    )
    movies_metadata = movies_metadata[["MovieID", "Title"]]
    movies_metadata["MovieID"] = movies_metadata["MovieID"].astype(int)
    
    return top_100_movies, similarity_matrix, movies_metadata

top_100_movies, similarity_matrix, movies_metadata = load_data()

def get_poster_url(movie_id):
    base_url = "https://liangfgithub.github.io/MovieImages/"
    poster_url = f"{base_url}{movie_id}.jpg?raw=true"
    return poster_url

top_100_movies_with_titles = top_100_movies.merge(movies_metadata, on="MovieID", how="left")

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

st.title("Movie Recommendation System")
st.write("Rate movies below, and get personalized recommendations!")

# Step 1: Display the top 100 movies for rating
st.subheader("Please rate the following movies:")
user_ratings = pd.Series(index=top_100_movies_with_titles["MovieID"], dtype=float)

# Use columns to display movies in a grid of 5 * 20
cols = st.columns(5)
for idx, row in top_100_movies_with_titles.iterrows():
    movie_id = row["MovieID"]
    movie_title = row["Title"]
    with cols[idx % 5]:
        st.image(get_poster_url(movie_id), caption=movie_title, width=120)
        rating = st.slider(f"Rate '{movie_title}'", 1, 5, value=0, step=1, key=f"rating_{movie_id}")
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
        recommended_movies = movies_metadata[movies_metadata["MovieID"].isin(recommendations.index)]
        for idx, row in recommended_movies.iterrows():
            st.image(get_poster_url(row["MovieID"]), caption=row["Title"], width=120)
            st.write(f"#{idx + 1}: {row['Title']} (Predicted rating: {recommendations[row['MovieID']]:.2f})")