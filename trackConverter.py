import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

types='test training validation'.split()
genres = 'Classical Electronic Pop Rock Hip-Hop'.split()
working_directory = os.path.dirname(os.getcwd())
spect_trimmed_len = 320

counter = 0
for t in types:
    mylist=[]
    label=[]
    i=0
    for g in genres:
        tmplabel=[0,0,0,0,0]
        tmplabel[i]=1
        i=i+1
        genre_path = os.path.join(working_directory, "ML", t, g)
        for filename in os.listdir(genre_path):
            y,sr=librosa.load(os.path.join(genre_path, filename))
            print(filename+ " - " + g + " - " + str(counter))
            spect=librosa.feature.melspectrogram(y=y,sr=sr,n_fft=4096, hop_length=2048)
            spect=librosa.power_to_db(spect,ref=np.max)
            if spect.shape[1] >= spect_trimmed_len:
                mylist.append(spect[:,:spect_trimmed_len])
                label.append(tmplabel)
            counter+=1
    spectrograms = np.array(mylist).astype(np.float32)
    labels = np.array(label)

    with open(f'{t}.npz', "wb") as fp:
        np.savez_compressed(fp, spectrograms=spectrograms) 
    with open(f'{t}_label.npz', "wb") as fp:
        np.savez_compressed(fp, labels=labels) 
