import os
import librosa
import csv
import numpy as np

song_duration = 10
directory = './wav_genres/'
csv_file = 'genre_data.csv'

def init():
    header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    data = []
    data.append(header)
    return data

def extract_features(directory, data):
    for genre in os.listdir(directory):
        print(genre)
        for song in os.listdir(f'{directory}{genre}'):
            song_path = f'{directory}{genre}/{song}'
            y, sr = librosa.load(song_path, mono=True, duration=song_duration)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            rmse = librosa.feature.rms(y)
            info = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                info += f' {np.mean(e)}'
            info += f' {genre}'
            info = info.split()
            data.append(info)
    return data

def write_data_into_file(data, file_name):
    with open(file_name, mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)
    
    
data = init()
data = extract_features(directory, data)
write_data_into_file(data, csv_file)

#towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8
#ismir2011.ismir.net/papers/OS6-1.pdf
#www-ai.cs.tu-dortmund.de/audio.html
#https://sfb876.tu-dortmund.de/PublicPublicationFiles/homburg_etal_2005a.pdf
#github.com/parulnith/Music-Genre-Classification-with-Python
#
