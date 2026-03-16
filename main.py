from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import cv2
import io
from groq import Groq
import requests 
from fastapi import UploadFile, File
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from functools import lru_cache
from fastapi.middleware.cors import CORSMiddleware
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import json
import base64

# -----------------------------
# Load Credentials and API
# -----------------------------

import os
from dotenv import load_dotenv

load_dotenv()



GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
api_key = os.getenv("API_KEY")

# Initialize app
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET),
                     requests_timeout=10)
groq_client = Groq(api_key=GROQ_API_KEY)
app = FastAPI(title="Emotion Detection (Camera)")

templates = Jinja2Templates(directory="templates")

# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load models
# -----------------------------
emotion_store = {"emotion": None}


DICT_EMO = [
    "Happiness",
    "Sadness",
    "Anger",
    "Surprise",
    "Fear",
    "Disgust",
    "Neutral"
]

# -----------------------------
# Load DataFrame
# -----------------------------

@lru_cache()
def load_movie_model():

    movies_df = pd.read_csv("movies_data.csv")
    movies_df = movies_df[movies_df["genres"].notna()]

    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(movies_df["combined"])

    return movies_df, tfidf_matrix

@lru_cache()
def load_book_model():

    books_df = pd.read_csv("books_dataset.csv")

    books_df = books_df.fillna({
        "authors": "Unknown",
        "thumbnail": "",
        "preview_link": ""
    })

    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(books_df["combined"])

    return books_df, tfidf_matrix

songs_df=pd.read_csv("./tracks.csv")
movies_df = pd.read_csv("./movies_data.csv")
# movies_df = movies_df[movies_df["genres"].notna()]
# cv = CountVectorizer()
# count_vectorizer_matrix = cv.fit_transform(movies_df["combined"])
# similarity_mat=cosine_similarity(count_vectorizer_matrix)

df_features_scaled=pd.read_csv("tracks_features_scaled.csv")

books_df = pd.read_csv("./books_dataset.csv")

# book_cv = CountVectorizer()
# book_count_vectorizer_matrix = book_cv.fit_transform(books_df["combined"])
# book_similarity_mat=cosine_similarity(book_count_vectorizer_matrix)
# -----------------------------
# Emotion inference
# -----------------------------

def predict_emotion(frame):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image and then to base64
    pil_img = Image.fromarray(frame_rgb)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    base64_face = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Prepare your prompt
    prompt = """
        Analyze the facial expression of the person in this image.

        Respond ONLY with a JSON object containing:
        1. "emotion": one of ["Happiness", "Sadness", "Anger", "Surprise", "Fear", "Disgust", "Neutral"]
        2. "confidence_reason": Short explanation (1-2 sentences)
        """

    try:
    # Send to Groq LLaMA Scout
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_face}"}
                        }
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            response_format={"type": "json_object"},
        )
    
        raw_content = response.choices[0].message.content
        parsed = json.loads(raw_content)
        # Extract fields
        emotion = parsed["emotion"]
        print(f"Predicted Emotion: {emotion}")
        confidence_reason = parsed["confidence_reason"]
        return emotion 
    
    except Exception as e:
        print("Groq error:", e)
        return "Neutral"
    
# -----------------------------
# Emotion Recommender Inference
# -----------------------------

# -----------------------------
# Song emotion Recommender
# -----------------------------
def recommend_songs(emotion):

    df = songs_df.sort_values(emotion).head(50)
    df =df.sample(15) # Randomly select 15 songs from the top 50

    results = []

    for track_id in df["id"]:
        try:
            track = sp.track(track_id)
            results.append({
                    "id": track["id"],
                    "name": track["name"],
                    "artist": ", ".join(a["name"] for a in track["artists"]),
                    "album": track["album"]["name"],
                    "image": track["album"]["images"][0]["url"] 
                                    if track.get("album") and track["album"].get("images") 
                                    else None,
                    "url": track["external_urls"]["spotify"],
                    "preview_url": track.get("preview_url")
                })
        except Exception as e:
            print(f"Error fetching track {track_id}: {e}")  
        
    return results

# -----------------------------
# Movies emotion Recommender
# -----------------------------
def recommend_movies(emotion):

    genre_map = {
        "Happiness": ["Comedy","Family","Romance"],
        "Sadness": ["Romance","Drama"],
        "Anger": ["Crime","Thriller"],
        "Fear": ["Thriller","Horror"],
        "Surprise": ["Mystery","Fantasy"],
        "Disgust": ["Crime","Drama"],
        "Neutral": ["Drama","Adventure"]
    }

    genres = genre_map.get(emotion, ["Drama"])

    mask = movies_df["genres"].apply(
        lambda x: any(g in x for g in genres)
    )
    df = movies_df.head(1000)  # Consider only the top 1000 movies for performance
    df = movies_df[mask].sample(15)

    return df[["title","genres", "id","imdb_id","cast","cast_orig","genres_orig"]].to_dict(orient="records")

# -----------------------
# BOOK emotion RECOMMENDER
# -----------------------
def recommend_books_fxn(emotion):
    emotion_category_map = {
    "Happiness": [
        "Fiction",
        "Comics & Graphic Novels",
        "Travel",
        "Cooking",
        "Sports & Recreation",
        "Family & Relationships",
        "Juvenile Fiction",
        "Young Adult Fiction"
    ],

    "Sadness": [
        "Biography & Autobiography",
        "Poetry",
        "Religion",
        "Philosophy",
        "Psychology",
        "Self-Help"
    ],

    "Disgust": [
        "Fantasy fiction",
        "Science fiction",
        "Comics & Graphic Novels",
        "Art",
        "Photography"
    ],

    "Anger": [
        "Philosophy",
        "Political Science",
        "Social Science",
        "History",
        "Ethics"
    ],

    "Fear": [
        "Self-Help",
        "Psychology",
        "Religion",
        "Health & Fitness",
        "Medical"
    ],

    "Neutral": [
        "Fiction",
        "Business & Economics",
        "Computers",
        "Science",
        "History",
        "Education"
    ],

    "Surprise": [
        "Science",
        "Artificial intelligence",
        "Technology & Engineering",
        "Astronomy",
        "Mathematics",
        "Computers"
    ]
    }
    categories = emotion_category_map.get(emotion, [])
    
    filtered = books_df[books_df["categories"].isin(categories)]
    
    filtered=filtered.sample(n=min(10, len(filtered)))
    
    return filtered.to_dict(orient="records")


# -----------------------
# PREFERENCE RECOMMENDER
# -----------------------
# -----------------------
# BOOK preference RECOMMENDER
# -----------------------
# Utility function to get track details from the dataframe
def get_track_details(track_name, df=songs_df):
    """Fetches the first matching track's details from the dataframe."""
    track_query = df.loc[df['name'].str.lower() == track_name.lower()]
    if not track_query.empty:
        return track_query.iloc[0]
    return None # Return None if not found

# --- 4.1. Content-Based Filtering Recommender (MEMORY OPTIMIZED) ---
def content_based_recommender(track_name, n_recommendations=50):
    """Recommends songs based on cosine similarity of audio features.
    This version is memory-efficient and avoids creating a full similarity matrix."""
    if songs_df.empty:
        print("Data not loaded. Cannot provide recommendations.")
        return pd.DataFrame()

    # Find the track
    track_details = get_track_details(track_name)
    if track_details is None:
        print(f"Track '{track_name}' not found in the dataset.")
        return pd.DataFrame()

    track_index = track_details.name # Get the index of the track
    track_features = df_features_scaled.loc[track_index].values.reshape(1, -1)

    # Calculate cosine similarity between the input track and all other tracks
    sim_scores = cosine_similarity(track_features, df_features_scaled)
    sim_scores = list(enumerate(sim_scores[0]))

    # Sort tracks by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar tracks (excluding the input track itself)
    sim_scores = sim_scores[1:n_recommendations+1]
    track_indices = [i[0] for i in sim_scores]
    recommendations = songs_df.iloc[track_indices]
    sorted_recs = recommendations.sort_values(by='popularity', ascending=False)
    df = sorted_recs.head(15)
    results = []

    for track_id in df["id"]:
        try:
            track = sp.track(track_id)
            results.append({
                    "id": track["id"],
                    "name": track["name"],
                    "artist": ", ".join(a["name"] for a in track["artists"]),
                    "album": track["album"]["name"],
                    "image": track["album"]["images"][0]["url"] 
                                    if track.get("album") and track["album"].get("images") 
                                    else None,
                    "url": track["external_urls"]["spotify"],
                    "preview_url": track.get("preview_url")
                })
        
        except Exception as e:
            print(f"Error fetching track {track_id}: {e}")
    return results

# -----------------------
# MOVIE PREFERENCE RECOMMENDER
# -----------------------
def movie_recommender(movie_title):

    movies_df, tfidf_matrix = load_movie_model()

    movie_title = movie_title.lower()

    matches = movies_df[
        movies_df["title"].str.lower() == movie_title
    ]

    if matches.empty:
        return []

    idx = matches.index[0]

    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    top_indices = sim_scores.argsort()[-11:-1][::-1]

    movie_array = []

    for i in top_indices:

        movie_array.append({
            "title": movies_df.iloc[i]["title"],
            "genres": movies_df.iloc[i]["genres_orig"],
            "cast": " ".join(
                str(movies_df.iloc[i]["cast_orig"]).split(",")[:3]
            ),
            "id": movies_df.iloc[i]["id"],
            "imdb_id": movies_df.iloc[i]["imdb_id"]
        })

    return movie_array

def book_recommender(book_title):

    books_df, tfidf_matrix = load_book_model()

    matches = books_df[
        books_df["title"].str.contains(book_title, case=False, na=False)
    ]

    if matches.empty:
        return []

    idx = matches.index[0]

    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    top_indices = sim_scores.argsort()[-11:-1][::-1]

    results = []

    for i in top_indices:

        results.append({
            "title": books_df.iloc[i]["title"],
            "authors": books_df.iloc[i]["authors"],
            "thumbnail": books_df.iloc[i]["thumbnail"],
            "preview": books_df.iloc[i]["preview_link"]
        })

    return results
# -----------------------
# ROUTES
# -----------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# -----------------------------
# EMOTION PAGE
# -----------------------------

@app.get("/emotion-page", response_class=HTMLResponse)
def emotion_page(request: Request):
    return templates.TemplateResponse("emotion.html", {"request": request})


# -----------------------------
# PREFERENCE PAGE
# -----------------------------

@app.get("/preference-page", response_class=HTMLResponse)
def preference_page(request: Request):
    return templates.TemplateResponse("preference.html", {"request": request})


# -----------------------------
# EMOTION DETECTION
# -----------------------------


@app.post("/detect-emotion")
async def detect_emotion(image: UploadFile = File(...)):

    img = await image.read()
    img = Image.open(io.BytesIO(img)).convert("RGB")
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    emotion = predict_emotion(frame)

    emotion_store["emotion"] = emotion

    return {"emotion": emotion}

# -----------------------
# Song RECOMMENDER
# -----------------------
@app.get("/search/songs")
def search_songs(query: str):

    if not query:
        return []

    matches = songs_df[
        songs_df["name"].str.contains(query, case=False, na=False)
    ].head(10)

    return matches["name"].tolist()

@app.get("/recommend/songs")
def recommend_songs_endpoint():
    emotion = emotion_store["emotion"]

    if not emotion:
        raise HTTPException(400, "Emotion not detected")

    # Get top 10 songs from Spotify using emotion
    # Use your existing recommend_songs() function logic
    results = recommend_songs(emotion)

    # Prepare proper JSON with image and Spotify URL
    songs_list = []
    for track in results:
        songs_list.append({
            "name": track["name"],
            "artist": track["artist"],
            "image": track["image"],          # album image
            "url": track["url"],              # Spotify link
            "preview": track.get("preview")   # optional preview URL
        })

    return songs_list

@app.get("/preference/from-track")
def recommend_from_track(track: str):

    results = content_based_recommender(track)

    songs_list = []

    for song in results:
        songs_list.append({
            "name": song["name"],
            "artist": song["artist"],
            "image": song["image"],
            "url": song["url"],
            "preview": song.get("preview_url")
        })

    return songs_list

# -----------------------------
# MOVIE RECOMMENDER
# -----------------------------
@app.get("/search/movies")
def search_movies(query: str):

    if not query:
        return []

    matches = movies_df[
        movies_df["title"].str.contains(query, case=False, na=False)
    ].head(10)

    return matches["title"].tolist()


@app.get("/recommend/movies")
def recommend_movies_endpoint():
    emotion = emotion_store["emotion"]

    if not emotion:
        raise HTTPException(400, "Emotion not detected")

    df = recommend_movies(emotion)  # get the list using your function
    

    movies_list = []
    for movie in df:
        tmdb_id= movie["id"]
        imdb_id = movie.get("imdb_id")
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}&language=en-US"
        response = requests.get(url)
        tmdb_data = response.json()
        movies_list.append({
            "title": movie["title"].title(),
            "genres": movie["genres_orig"],
            "cast":" ".join(movie.get("cast_orig","N/A").split(",")[:3]),  # Show top 3 cast members
            "tmdb_url": f"https://www.themoviedb.org/movie/{movie.get('id', '')}",
            "imdb_url": f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None,
            "poster": f"https://image.tmdb.org/t/p/w500{tmdb_data.get('poster_path')}" if tmdb_data.get("poster_path") else None
        })

    return movies_list

@app.get("/preference/from-movie")
def recommend_from_movie(movie: str):

    movies = movie_recommender(movie)

    movies_list = []

    for movie in movies:

        tmdb_id = movie["id"]
        imdb_id = movie.get("imdb_id")

        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}&language=en-US"

        response = requests.get(url)
        tmdb_data = response.json()

        movies_list.append({
            "title": movie["title"],
            "genres": movie["genres"],
            "cast": movie["cast"],  # Show top 5 cast members
            "tmdb_url": f"https://www.themoviedb.org/movie/{movie.get('id', '')}",
            "imdb_url": f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None,
            "poster": f"https://image.tmdb.org/t/p/w500{tmdb_data.get('poster_path')}" if tmdb_data.get("poster_path") else None
        })

    return movies_list



# -----------------------------
# BOOK RECOMMENDER
# -----------------------------

@app.get("/search/books")
def search_books(query: str):

    if not query:
        return []

    matches = books_df[
        books_df["title"].str.contains(query, case=False, na=False)
    ].head(10)

    return matches["title"].tolist()


@app.get("/recommend/books")
def recommend_books():
    emotion = emotion_store["emotion"]

    if not emotion:
        raise HTTPException(400, "Emotion not detected")

    df = recommend_books_fxn(emotion)  # get the list using your function
    

    book_list = []
    for book in df:
        book_list.append({
            "title": book["title"],
            
            "authors": book["authors"],

            "thumbnail": book["thumbnail"],
            "preview": book["preview_link"]
        })

    return book_list


@app.get("/preference/from-books")
def recommend_from_books(book: str):

    book_list = book_recommender(book)

    return book_list

