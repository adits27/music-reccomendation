import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from scipy.spatial.distance import cdist
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import time

## Initialize the Spotify Client
client_id = os.environ["SPOTIFY_CLIENT_ID"]
client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

## Load the data
dataset = pd.read_csv("data/data.csv")

# Calculate the sum of squared distances for different values of k
X = dataset.select_dtypes(np.number)
X.drop(['year'],axis=1,inplace=True)
columns = list(X.columns)
sse = []
for k in range(1, 25):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(range(1, 25), sse, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances (SSE)')
plt.grid()
plt.show()

## Instantiate Pipeline with optimal cluster number = 7
pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=7))])
pipeline.fit(X)

def find_song(name):
    song_data = defaultdict()
    results = sp.search(q= 'track: {}'.format(name), limit=1)
    if results['tracks']['items'] == []:
        return None
    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]
    song_data['name'] = [name]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]
    for key, value in audio_features.items():
        song_data[key] = value
    return pd.DataFrame(song_data)

def get_song(song, data):
    try:
        song_data = data[(data['name'] == song['name'])].iloc[0]
        data = song_data
    except IndexError:
        data = find_song(song['name'])
    uri_link = "https://open.spotify.com/embed/track/" + data['id'] + "?utm_source=generator&theme=0"
    components.iframe(uri_link, height=80)
    return data
    
def get_vector(song, spotify_data):
    song_data = get_song(song[0], spotify_data)
    if song_data is None:
        print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
    song_vector = song_data[columns].values
    return song_vector

def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict

def recommend(song, spotify_data, n_songs=10):
    metadata_cols = ['name', 'artists','id']
    song_dict = flatten_dict_list(song)
    song_vector = get_vector(song, spotify_data)
    scaler = pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[columns])
    scaled_song_vector = scaler.transform(song_vector.reshape(1, -1))
    ## Cosine Similarity
    distances = cdist(scaled_song_vector, scaled_data, 'cosine') 
    index = list(np.argsort(distances)[:, :n_songs][0])
    print(index)
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')


st.title("Spotify Recommendation System")
col,col2=st.columns([1,3])
input = col.text_input("Enter the Song Name")
number = col2.slider('How many recommendations would you like?', 0, 15, 1)
button = st.button("Get Recommendations")

if button:
        recommendations = recommend([{'name':input }],  dataset, number+1)
        time.sleep(2)
        st.subheader(f'Top `{number}` recommendations for `{input}`')
        print(recommendations)
        for i in recommendations:
            uri_link = "https://open.spotify.com/embed/track/" + i['id'] + "?utm_source=generator&theme=0"
            components.iframe(uri_link, height=80)
            time.sleep(1)