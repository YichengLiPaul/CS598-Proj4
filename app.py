import streamlit as st
import pandas as pd
import numpy as np

# File URLs on GitHub
TOP_MOVIES_URL = "https://raw.githubusercontent.com/YichengLiPaul/CS598-Proj4/main/top_movies.txt"
SIMILARITY_MATRIX_URL = "https://raw.githubusercontent.com/YichengLiPaul/CS598-Proj4/main/top_30_similarity_matrix.csv"
MOVIES_METADATA_URL = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Load data
@st.cache_data
def load_data():
    # Load all movies (MovieID list)
    all_movies = pd.read_csv(TOP_MOVIES_URL, header=None, names=["MovieID"])
    all_movies["MovieID"] = all_movies["MovieID"].str.replace("m", "", regex=False).astype(int)
    
    # Load similarity matrix
    similarity_matrix = pd.read_csv(SIMILARITY_MATRIX_URL, index_col=0)
    similarity_matrix.columns = similarity_matrix.columns.str.replace("m", "", regex=False).astype(int)
    similarity_matrix.index = similarity_matrix.index.str.replace("m", "", regex=False).astype(int)
    
    # Load movies.dat for metadata
    movies_metadata = pd.read_csv(
        MOVIES_METADATA_URL, sep="::", header=None, engine="python", encoding="latin1",
        names=["MovieID", "Title", "Genres"]
    )
    movies_metadata = movies_metadata[["MovieID", "Title"]]
    movies_metadata["MovieID"] = movies_metadata["MovieID"].astype(int)
    
    # Merge all movies with metadata
    all_movies = all_movies.merge(movies_metadata, on="MovieID", how="left")
    
    # Select the top 250 movies and randomize 100 for review
    top_250_movies = all_movies.head(250)
    top_100_movies = top_250_movies.sample(100, random_state=np.random.randint(0, 10000))
    
    return all_movies, top_100_movies, similarity_matrix, movies_metadata

# Load the data
all_movies, top_100_movies, similarity_matrix, movies_metadata = load_data()

# Function to get poster URL
def get_poster_url(movie_id):
    base_url = "https://liangfgithub.github.io/MovieImages/"
    poster_url = f"{base_url}{movie_id}.jpg?raw=true"
    return poster_url

# myIBCF function
def myIBCF(new_user, similarity_matrix, all_movies):
    """
    Item-Based Collaborative Filtering Recommendation Function.
    """
    # Ensure new_user has all movies (fill missing movies with NaN)
    new_user_full = pd.Series(index=similarity_matrix.columns, dtype=float)
    new_user_full.update(new_user)  # Update with ratings provided by the user
    
    # Initialize predictions
    predictions = pd.Series(index=similarity_matrix.columns, dtype=float)
    
    # Predict ratings
    for movie in similarity_matrix.columns:
        if pd.isna(new_user_full[movie]):  # Only predict for unrated movies
            similar_movies = similarity_matrix.loc[movie].dropna()
            rated_movies = similar_movies.index.intersection(new_user_full.dropna().index)
            if len(rated_movies) > 0:
                weights = similar_movies[rated_movies]
                ratings = new_user_full[rated_movies]
                numerator = (weights * ratings).sum()
                denominator = weights.sum()
                if denominator > 0:
                    predictions[movie] = numerator / denominator
    
    # Get the top 10 recommended movies
    sorted_predictions = predictions.dropna().sort_values(ascending=False)
    top_recommendations = sorted_predictions.head(10)
    
    # Append movies from all_movies if fewer than 10 recommendations
    if len(top_recommendations) < 10:
        # Movies the user hasn't rated yet
        unrated_movies = all_movies[~all_movies["MovieID"].isin(new_user.dropna().index)]
        additional_movies = unrated_movies["MovieID"].head(10 - len(top_recommendations))
        for movie_id in additional_movies:
            top_recommendations.loc[movie_id] = np.nan  # Placeholder score
    
    # Merge recommendations with movie titles
    recommended_movies = pd.DataFrame({
        "MovieID": top_recommendations.index,
        "Predicted_Rating": top_recommendations.values
    }).merge(movies_metadata, on="MovieID", how="left")
    
    return recommended_movies

# Streamlit UI
st.title("Movie Recommendation System")
st.write("Rate movies below, and get personalized recommendations!")

# Step 1: Display the first 100 movies for rating
st.subheader("Please rate the following movies:")
user_ratings = pd.Series(index=top_100_movies["MovieID"], dtype=float)

# Use columns to display movies in a grid
cols = st.columns(5)  # 5 movies per row
for idx, row in top_100_movies.iterrows():
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
    recommendations = myIBCF(user_ratings, similarity_matrix, all_movies)
    if recommendations.empty:
        st.write("No recommendations found. Please rate more movies!")
    else:
        for idx, row in recommendations.iterrows():
            st.write(f"#{idx + 1}: {row['Title']}")
            st.image(get_poster_url(row["MovieID"]), width=120)
            if not pd.isna(row["Predicted_Rating"]):
                st.write(f"Predicted rating: {row['Predicted_Rating']:.2f}")
            else:
                st.write(f"Popular movie fallback")
