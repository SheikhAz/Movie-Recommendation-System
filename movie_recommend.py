import pandas as pd
import pickle
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import time
import re
import base64


# ------------------ Load data ------------------ #
Data = pickle.load(open("movies_dict.pkl", mode="rb"))
movies = pd.DataFrame(Data)
Similarity = pickle.load(open("similarity.pkl", mode="rb"))

# TMDB config
TMDB_API_KEY = "de5225fd4c365705ec32b035b54cbb84"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


# ------------------ Movie Detail Fetcher ------------------ #
def fetch_movie_details(query):
    url = f"{TMDB_BASE_URL}/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": query}
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data["results"]:
                    return data["results"][0]
        except requests.exceptions.RequestException:
            time.sleep(2)
    return None


# Convert image to base64
def get_base64_bg(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"


# Load background image
bg_img = get_base64_bg("Background.jpg")  # Ensure this file is in the same directory

# Set the background using Streamlit's HTML injection
st.markdown(
    f"""
    <style>
        html, body, [data-testid="stApp"] {{
            height: 100%;
            width: 100%;
            background-image: url("{bg_img}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center center;
            background-attachment: fixed;
            overflow-x: hidden;
        }}

        /* Optional: Add slight overlay for contrast */
        [data-testid="stApp"]::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: rgba(0, 0, 0, 0.4);  /* Adjust transparency here */
            z-index: -1;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def get_poster_url(movie_title):
    movie_details = fetch_movie_details(movie_title)
    if movie_details and "poster_path" in movie_details:
        return f"{TMDB_IMAGE_BASE_URL}{movie_details['poster_path']}"
    return "https://via.placeholder.com/500x750?text=Poster+Not+Available"


# Make sure Tags are lowercase and valid
movies["Tags"] = movies["Tags"].apply(lambda x: x.lower() if isinstance(x, str) else "")

# Extract all tag words
tag_words = set()
movies["Tags"].apply(lambda tag: tag_words.update(re.findall(r"\w+", tag)))


def chatbot_tag_matcher_with_scoring(user_input):
    user_words = re.findall(r"\w+", user_input.lower())
    match_scores = []

    for idx, row in movies.iterrows():
        movie_tags = set(re.findall(r"\w+", row["Tags"]))
        match_count = sum(1 for word in user_words if word in movie_tags)
        if match_count > 0:
            match_scores.append((idx, match_count))

    # Sort by match count
    sorted_matches = sorted(match_scores, key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in sorted_matches[:5]]

    return movies.iloc[top_indices] if top_indices else pd.DataFrame()


# ------------------ Recommender Logic ------------------ #
def recommend(movie, genre=None):
    if movie not in movies["movie_name"].values:
        st.error("Selected movie not found in the database.")
        return [], []

    movie_index = movies[movies["movie_name"] == movie].index[0]
    distances = Similarity[movie_index]
    sorted_movies = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)

    recommended_movies = []
    recommended_posters = []

    for i in sorted_movies:
        similar_movie = movies.iloc[i[0]]
        if similar_movie["movie_name"] == movie:
            continue
        if genre and genre not in similar_movie["genre"]:
            continue

        recommended_movies.append(similar_movie["movie_name"])
        recommended_posters.append(get_poster_url(similar_movie["movie_name"]))

        if len(recommended_movies) == 5:
            break

    return recommended_movies, recommended_posters


# ------------------ Streamlit UI ------------------ #
st.markdown(
    """
    <h1 style="
        background: linear-gradient(to right, #ff6a00, #ee0979);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
    ">
        Movie Recommendation System
    </h1>
    """,
    unsafe_allow_html=True,
)


# Ensure genres are lists
movies["genre"] = movies["genre"].apply(lambda x: x if isinstance(x, list) else eval(x))

# Movie selection dropdown
Selected_Movie = st.selectbox(
    "Choose a movie you like:", sorted(movies["movie_name"].unique())
)

# Genre filter
all_genres = set()
movies["genre"].apply(lambda genres: all_genres.update(genres))
Selected_Genre = st.selectbox("Optional: Filter by genre", [""] + sorted(all_genres))

# Recommend Button centered using columns
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    recommend_clicked = st.button("Recommend")


if recommend_clicked:
    with st.spinner("Finding the best matches..."):
        genre_filter = Selected_Genre if Selected_Genre else None
        Top_5_Movies, Top_5_Posters = recommend(Selected_Movie, genre_filter)

    if Top_5_Movies:
        st.subheader("Top 5 Recommendations:")
        cols = st.columns(len(Top_5_Movies))
        for col, movie_title, poster_url in zip(cols, Top_5_Movies, Top_5_Posters):
            movie_details = fetch_movie_details(movie_title)
            if movie_details:
                overview = movie_details.get("overview", "No description available.")[
                    :150
                ]
                rating = movie_details.get("vote_average", "N/A")
                release_date = movie_details.get("release_date", "N/A")

                with col:
                    st.image(poster_url, use_container_width=True)
                    with st.expander(movie_title):
                        st.markdown(f"**Rating:** {rating}")
                        st.markdown(f"**Release Date:** {release_date}")
                        st.markdown(f"*{overview}...*")

import streamlit as st

# Inject custom CSS for gradient text
st.markdown(
    """
    <style>
    .gradient-text {
        font-size: 28px;
        font-weight: 600;
        background: linear-gradient(90deg, #ff4b1f, #1fddff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.5em 0;
    }
    hr {
        border: none;
        border-top: 2px solid #bbb;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Horizontal line
st.markdown("---")

# Gradient subheader
st.markdown(
    '<div class="gradient-text">🎬 MovieBot: Describe the kind of movie you\'re in the mood for!</div>',
    unsafe_allow_html=True,
)

# Chat input
user_prompt = st.chat_input("Type something like 'funny space adventure with a twist'")

if user_prompt:
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.spinner("Thinking..."):
        filtered_movies = chatbot_tag_matcher_with_scoring(user_prompt)

    with st.chat_message("assistant"):
        if not filtered_movies.empty:
            st.write("Here are some great movie matches based on your vibe:")

            sample_movie = filtered_movies.iloc[0]["movie_name"]
            top_movies, top_posters = recommend(sample_movie)

            cols = st.columns(len(top_movies))
            for col, title, poster in zip(cols, top_movies, top_posters):
                with col:
                    st.image(poster, use_container_width=True)
                    st.caption(title)
        else:
            st.write(
                "Hmm, I couldn’t find anything based on that description. Try keywords like 'romantic', 'thriller', or 'funny'."
            )
