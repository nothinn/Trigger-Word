import soundfile as sf
import xml.etree.ElementTree as ET
import os

import numpy as np

from pathlib import Path
import librosa

words = dict()
#    data, samplerate = sf.read(audioPath)
filesr = 33100 #SampleRate for output file

dirs = os.listdir("english/")

Path("words/").mkdir(parents=True, exist_ok=True)
for dir in dirs:
    print(dir)

    path = "english/" + dir
    try:

        audioPath = path + "/audio.ogg"

        tree = ET.parse(path + "/aligned.swc")



        print("Adding: {}".format(audioPath))
        data, samplerate = librosa.load(audioPath,filesr)
        print(samplerate)


        root = tree.getroot()
        for n in root.findall(".//n"): #Finds all "n" elements in entire tree
            word = n.get('pronunciation').lower()
            start = n.get('start')
            end = n.get('end')
            if start: #These must not be None
                if end:

                    
                    Path("words/" + word).mkdir(parents=True, exist_ok=True)

                    try:
                        info = open("words/" + word + "/list.txt","r")
                        lines = info.readlines()
                        #print(lines)
                        index = 0
                        if(len(lines) >= 1):
                            lastLine = lines[-1]
                            #print(lastLine)
                            index = int(lastLine.split(" ")[0]) + 1
                        info.close()
                    except FileNotFoundError:
                        index = 0
                        pass
                    
                    tempPath = "words/" + word + "/" + word + "_" + str(index) +".wav"
                    #print(tempPath)
                    with sf.SoundFile(tempPath, "w", samplerate=filesr,channels=1) as f:                  

                        startSample = samplerate * (int(start) / 1000.0 + 0.1) #.1 second before and after
                        endSample = samplerate * (int(end) / 1000.0 + 0.1)

                        info = open("words/" + word + "/list.txt","a")
                        info.write(str(index))
                        info.write(" ")
                        info.write(audioPath)
                        info.write(" ")
                        info.write(str(startSample))
                        info.write(" ")
                        info.write(str(endSample))
                        info.write("\n")
                        info.flush()
                        info.close()
                        
                        f.write(data[int(startSample):int(endSample)])

                    try:
                        words[word].append((audioPath, int(start),int(end)))
                    except KeyError:
                        words[word] = [(audioPath,int(start),int(end))]
    except Exception as e:
        print("{} failed".format(path))
        print(e)


print("Went through all of Wikipedia")
print()

'''
for (word,lst) in sorted(words.items(),key = lambda kv:(len(kv[1]), kv[0])):
    try:
        print("Word: {}, elements: {}".format(word,len(lst)))

        Path("words/" + word).mkdir(parents=True, exist_ok=True)

        lastPath = ""

        info = open("words/" + word + "/list.txt","w")


        #Take the word "later" and make a file of that:
        for (index, element) in enumerate(lst):
            #try:
            tempPath = "words/" + word + "/" + word + "_" + str(index) +".wav"
            print(tempPath)
            with sf.SoundFile(tempPath, "w", samplerate=filesr,channels=1) as f:
                if lastPath != element[0]:
                    lastPath = element[0]
                    print("Adding: {}".format(element[0]))
                    data, samplerate = librosa.load(element[0],filesr)
                    print(samplerate)

                

                startSample = samplerate * (element[1] / 1000.0 + 0.1) #.1 second before and after
                endSample = samplerate * (element[2] / 1000.0 + 0.1)

                info.write(str(index))
                info.write(" ")
                info.write(element[0])
                info.write(" ")
                info.write(str(element[1]))
                info.write(" ")
                info.write(str(element[2]))
                info.write("\n")
                
                
                
                f.write(data[int(startSample):int(endSample)])
           ### except Exception as e:
                #print("Exception at: " + element[0])
                #print(e)
                #pass

    except UnicodeEncodeError:
        pass
    finally:
        info.close()


'''
