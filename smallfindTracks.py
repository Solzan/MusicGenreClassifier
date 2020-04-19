import os
import shutil
import numpy as np
import pandas as pd
import IPython.display as ipd



def createGenreDirectory(newpath):
    if not os.path.exists(newpath+"\Classical"):
        os.makedirs(newpath+"\Classical")
    if not os.path.exists(newpath+"\Electronic"):
        os.makedirs(newpath+"\Electronic")
    if not os.path.exists(newpath+"\Pop"):
        os.makedirs(newpath+"\Pop")
    if not os.path.exists(newpath+"\Rock"):
        os.makedirs(newpath+"\Rock")
    if not os.path.exists(newpath+"\Rap"):
        os.makedirs(newpath+"\Rap")

def createSplitDirectory(newpath):
    if not os.path.exists(os.path.join(newpath+"\\training")):
        os.makedirs(os.path.join(newpath+"\\training"))
        createGenreDirectory(os.path.join(newpath+"\\training"))
    if not os.path.exists(os.path.join(newpath+"\\test")):
        os.makedirs(os.path.join(newpath+"\\test"))
        createGenreDirectory(os.path.join(newpath+"\\test"))
    if not os.path.exists(os.path.join(newpath+"\\validation")):
        os.makedirs(os.path.join(newpath+"\\validation"))
        createGenreDirectory(os.path.join(newpath+"\\validation"))
        

def findMusic(filePath,newPath,audioDir,data,splt):
    for track_id, genre in data.iteritems():
        tid_str = '{:06d}'.format(track_id)
        in_path=os.path.join(audioDir, tid_str[:3], tid_str + '.mp3')
        sl=splt[track_id]
        if(genre in ['Pop','Rap','Rock','Electronic','Classical']):
            out_path=newPath+"\\"+splt[track_id]+"\\"+genre
            sutil.move(in_path,out_path)
                


            
filePath=r"C:\Users\DellM4600\Desktop\fma_metadata\tracks.csv" # parh to tracks.csv
newPath=r"C:\Users\DellM4600\Desktop\ML"                       # parh to result directory
audioDir=r"C:\Users\DellM4600\Desktop\fma_small"               # parh to fma dataset
createSplitDirectory(newPath)


tracks = pd.read_csv(filePath, index_col=0, header=[0, 1])
small = (tracks['set', 'subset']=='small')
splt=tracks['set','split']
data = tracks.loc[small, ('track', 'genre_top')]
findMusic(filePath,newPath,audioDir,data,splt)


                        
    

            
            
            





