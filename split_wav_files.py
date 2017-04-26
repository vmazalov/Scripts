import argparse
import json
import os
import subprocess
from subprocess import call
import ntpath

parser = argparse.ArgumentParser(description="Read JSON file with wav files description and trim the KW from front")
parser.add_argument('--jsonFile', help='', required=False)
parser.add_argument('--outputFolder', help='', required=True)
args = parser.parse_args()

textout = ""
with open(args.jsonFile) as json_data:
    d = json.load(json_data)
    files = d['files']
    for e in files:
        wavFilePath =  e['wav']
        wavFileName = ntpath.basename(wavFilePath)
        if len(e['detections']) != 1:
            continue
        kwEnd = e['detections'][0]['end'] + 0.02
        cmd = "..\\path_to\\sox.exe " + wavFilePath + " " + os.path.join(args.outputFolder, wavFileName) + " trim " + str(kwEnd)
        print (cmd)

