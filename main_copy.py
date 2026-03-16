from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import cv2
from groq import Groq
import time
from fastapi import UploadFile, File
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from models import ResNet50, LSTMPyTorch, pth_processing
from fastapi.middleware.cors import CORSMiddleware
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Initialize Spotipy
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET))

app = FastAPI(title="Emotion Detection (Camera)")

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# GLOBAL EMOTION STORE
# -----------------------
emotion_store = {"emotion": None}

# -----------------------------
# Load models
# -----------------------------
face_detector = YOLO("weights/yolov8n-face.pt")

pth_backbone_model = ResNet50(7, channels=3)
pth_backbone_model.load_state_dict(
    torch.load("weights/FER_static_ResNet50_AffectNet.pt", map_location="cpu")
)
pth_backbone_model.eval()

pth_LSTM_model = LSTMPyTorch()
pth_LSTM_model.load_state_dict(
    torch.load("weights/FER_dinamic_LSTM_Aff-Wild2.pt", map_location="cpu")
)
pth_LSTM_model.eval()

DICT_EMO = {
    0: 'Neutral',
    1: 'Happiness',
    2: 'Sadness',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger'
}

songs_df=pd.read_csv("./tracks.csv")

scaler = MinMaxScaler()
songs_df[['valence','energy','tempo','danceability']] = scaler.fit_transform(
    songs_df[['valence','energy','tempo','danceability']]
)

emotion_targets = {
    0: {'valence': 0.5, 'energy': 0.5, 'tempo': 100, 'mode': 1, 'danceability': 0.5},  # Neutral
    1: {'valence': 0.85,'energy':0.8, 'tempo':120, 'mode':1, 'danceability':0.8},      # Happiness
    2: {'valence':0.2, 'energy':0.25, 'tempo':75, 'mode':0, 'danceability':0.25},      # Sadness
    3: {'valence':0.7, 'energy':0.8, 'tempo':140, 'mode':1, 'danceability':0.7},       # Surprise
    4: {'valence':0.25, 'energy':0.85, 'tempo':120, 'mode':0, 'danceability':0.45},    # Fear
    5: {'valence':0.15, 'energy':0.65, 'tempo':105, 'mode':0, 'danceability':0.45},    # Disgust
    6: {'valence':0.15, 'energy':0.9, 'tempo':145, 'mode':0, 'danceability':0.6}       # Anger
}

# -----------------------------
# Camera capture
# -----------------------------
def capture_from_camera(delay_sec=5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    time.sleep(delay_sec)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError("Failed to capture frame")

    return frame

def get_songs_by_mood(mood):
    try:
        results = sp.search(q=mood, type="track", limit=50)
        songs = []
        for item in results['tracks']['items']:
            song = {
                'name': item['name'],
                'artist': ", ".join(artist['name'] for artist in item['artists']),
                'album_image_url': item['album']['images'][0]['url'] if item['album']['images'] else None,
                'spotify_url': item['external_urls']['spotify'] if 'external_urls' in item else None
            }
            songs.append(song)
        return songs
    except Exception as e:
        return []

def get_songs_from_dataframe(df, emotion_id, mood):

    # Convert mood string → emotion_id
    emotion_id = next(
        (key for key, value in DICT_EMO.items() if value.lower() == mood.lower()),
        None
    )

    if emotion_id is None:
        return []

    target = emotion_targets[emotion_id]

    # Scale tempo properly (IMPORTANT FIX)
    tempo_scaled = scaler.transform(
        [[0, 0, target['tempo'], 0]]
    )[0][2]

    target_vector = np.array([
        target['valence'],
        target['energy'],
        tempo_scaled,
        target['mode'],
        target['danceability']
    ])

    # Calculate distance
    df = df.copy()  # avoid modifying global dataframe

    def distance(row):
        song_vector = np.array([
            row['valence'],
            row['energy'],
            row['tempo'],
            row['mode'],
            row['danceability']
        ])
        return np.linalg.norm(target_vector - song_vector)

    df['distance'] = df.apply(distance, axis=1)

    # Get best matches
    df = df.sort_values('distance').head(20)

    songs = []

    try:
        for track_id in df["id"].tolist():

            track = sp.track(track_id)

            song = {
                "id": track["id"],
                "name": track["name"],
                "artist": ", ".join(a["name"] for a in track["artists"]),
                "album": track["album"]["name"],
                "album_image_url": track["album"]["images"][0]["url"] 
                                if track.get("album") and track["album"].get("images") 
                                else None,
                "spotify_url": track["external_urls"]["spotify"],
                "preview_url": track.get("preview_url")
            }

            songs.append(song)

        return songs

    except Exception as e:
        print("Spotify Error:", e)
        return []    

# Function to get albums by mood using Spotify API
def get_albums_by_mood(mood):
    try:
        results = sp.search(q=mood, type="album", limit=20)
        albums = []
        for item in results['albums']['items']:
            album = {
                'name': item['name'],
                'artist': ", ".join(artist['name'] for artist in item['artists']),
                'image_url': item['images'][0]['url'] if item['images'] else None,
                'spotify_url': item['external_urls']['spotify'] if 'external_urls' in item else None
            }
            albums.append(album)
        return albums
    except Exception as e:
        return []
# -----------------------------
# Emotion inference
# -----------------------------
def predict_emotion(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    results = face_detector(rgb, conf=0.5, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None

    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face = rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None

    face = pth_processing(Image.fromarray(face))
    features = torch.relu(
        pth_backbone_model.extract_features(face)
    ).detach().numpy()

    lstm_f = torch.from_numpy(np.vstack([features] * 10)).unsqueeze(0)
    output = pth_LSTM_model(lstm_f).detach().numpy()

    cl = int(np.argmax(output))
    return {
        "emotion": DICT_EMO[cl],
        "confidence": round(float(output[0][cl]) * 100, 2)
    }

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/emotion/camera")
def emotion_from_browser(image: UploadFile = File(...)):

    img = Image.open(image.file).convert("RGB")
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    result = predict_emotion(frame)

    if result is None:
        return {"message": "No face detected"}

    return result

@app.get("/recommend/{mood}/{rec_type}")
def recommend_music(mood: str, rec_type: str):

    if rec_type == "song":
        results = get_songs_from_dataframe(songs_df, None, mood)
        return {"type": "song", "results": results}

    elif rec_type == "album":
        results = get_albums_by_mood(mood)
        return {"type": "album", "results": results}

    else:
        raise HTTPException(status_code=400, detail="Invalid recommendation type")
    