import os
import shutil
import numpy as np
import pandas as pd
import IPython.display as ipd



def createGenreDirectory(newpath):
    if not os.path.exists(os.path.join(newpath, "Classical")):
        os.makedirs(os.path.join(newpath, "Classical"))
    if not os.path.exists(os.path.join(newpath, "Electronic")):
        os.makedirs(os.path.join(newpath, "Electronic"))
    if not os.path.exists(os.path.join(newpath, "Pop")):
        os.makedirs(os.path.join(newpath, "Pop"))
    if not os.path.exists(os.path.join(newpath, "Rock")):
        os.makedirs(os.path.join(newpath, "Rock"))
    if not os.path.exists(os.path.join(newpath, "Hip-Hop")):
        os.makedirs(os.path.join(newpath, "Hip-Hop"))

def createSplitDirectory(newpath):
    if not os.path.exists(os.path.join(newpath, "training")):
        os.makedirs(os.path.join(newpath , "training"))
        createGenreDirectory(os.path.join(newpath, "training"))
    if not os.path.exists(os.path.join(newpath ,"test")):
        os.makedirs(os.path.join(newpath, "test"))
        createGenreDirectory(os.path.join(newpath, "test"))
    if not os.path.exists(os.path.join(newpath, "validation")):
        os.makedirs(os.path.join(newpath, "validation"))
        createGenreDirectory(os.path.join(newpath, "validation"))
        

def findTracks(filePath,newPath,audioDir,data,splt):
    for track_id, genre in data.iteritems():
        tid_str = '{:06d}'.format(track_id)
        in_path=os.path.join(audioDir, tid_str[:3], tid_str, ".mp3")
        sl=splt[track_id]
        if(genre in ['Pop','Hip-Hop','Rock','Electronic','Classical']):
            out_path=os.path.join(newPath, splt[track_id], genre)
            if(genre=='Rock' or genre=='Electronic'):
                if(np.random.random_sample()<0.5):
                    shutil.copy2(in_path,out_path)
            else:
                shutil.copy2(in_path,out_path)
                

working_directory = os.path.dirname(os.getcwd())

filePath = os.path.join(working_directory, "fma_metadata", "tracks.csv") # parh to tracks.csv
newPath = os.path.join(working_directory, "ML")                     # parh to result directory
audioDir = os.path.join(working_directory, "fma_large")               # parh to fma dataset
createSplitDirectory(newPath)

          

tracks = pd.read_csv(filePath, index_col=0, header=[0, 1])
small = (tracks['set', 'subset']!='')
splt=tracks['set','split']
data = tracks.loc[small, ('track', 'genre_top')]
findTracks(filePath,newPath,audioDir,data,splt)


                        
    

            
            
            





