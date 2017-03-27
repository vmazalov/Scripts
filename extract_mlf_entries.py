import os
import argparse

parser = argparse.ArgumentParser(description="Given SCP and MLF speech files, output a new MLF file with only those labels from the original MLF that have corresponding entry in SCP file.")
parser.add_argument('--inputSCP', help='', required=True)
parser.add_argument('--inputMLF', help='', required=False)
parser.add_argument('--outputMLF', help='', required=True)
args = parser.parse_args()

if not os.path.exists(args.inputSCP) or not os.path.exists(args.inputMLF):
    print('Input file is invalid.')

scpEntries = set()
with open(args.inputSCP) as f:
     for scpLine in f:
        scpEntries.add(scpLine[0:scpLine.find('.')])

mlfEntry = ""
mlfOut = ""
mlfIn = False
with open(args.inputMLF) as f:
     for mlfLine in f:
         if mlfLine.startswith('"'):
             mlfEntry = mlfLine[1:mlfLine.find('.')]
             if scpEntries.__contains__(mlfEntry):
                 mlfIn = True
             else:
                 mlfIn = False

         if mlfIn:
             mlfOut += mlfLine

with open(args.outputMLF, 'w') as f:
    f.write(mlfOut)
