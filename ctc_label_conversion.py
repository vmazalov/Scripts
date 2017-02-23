import os
import argparse

def adaptScp(scpFile, output):
    result = {}
    counter = 0
    with open(output, "w") as outputFile:
        with open(scpFile) as f:
            for line in f:
                splitted = line.split("=")
                if len(splitted) != 2:
                    raise Exception("Invalid input '%s', line with invalid format '%s'" % (scpFile, line))
                key = splitted[0].strip()
                if key.endswith(".mfc"):
                    key = key[:-4]
                newKey = str(counter)
                result[key] = newKey
                outputLine = newKey + "=" + splitted[1]
                outputFile.write(outputLine)
                counter += 1
    return result

def convertMlf(mlfFile, phoneList, output, keys):
    phoneMap = {}
    counter = 0
    with open(phoneList) as phoneFile:
        for phone in phoneFile:
            phoneMap[phone.strip()] = counter
            counter += 1
    previous = ""
    with open(output, "w") as outputFile:
        with open(mlfFile) as inputFile:
            for originalLine in inputFile:
                line = originalLine.strip()
                if previous == "." or previous == "#!MLF!#":
                    line = line.strip('"')
                    if line.endswith(".lab"):
                        line = line[:-4]

                    if line in keys:
                        seqId = keys[line]
                else:
                    parts = line.split(" ")
                    if parts.__len__() > 2:
                        startInd = int(parts[0].strip()) / 100000
                        endInd = int(parts[1].strip()) / 100000
                        phone = parts[2]
                        phoneInd = phoneMap[phone]
                        outputFile.write(seqId + " |l " + str(phoneInd) + ":2\n")
                        outLine = seqId + " |l " + str(phoneInd) + ":1\n"
                        for i in range(int(startInd)+1,int(endInd)):
                            outputFile.write(outLine)

                previous = line.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapts mlf and scp files to use logical keys of smaller length and converts MLF file to format for CNTK Text reader")
    parser.add_argument('--inputScpFile', help='Input scp file.', required=True)
    parser.add_argument('--inputMlfFile', help='Input mlf file that corresponds to the scp file.', required=True)
    parser.add_argument('--inputPhoneListFile', help='Input file with phone list.', required=True)
    parser.add_argument('--outputScpFile', help='Output scp file.', required=True)
    parser.add_argument('--outputLabelFile', help='Output label file.', required=True)
    args = parser.parse_args()
    
    if not os.path.exists(args.inputScpFile):
        print('Input scp file is invalid.')

    if not os.path.exists(args.inputMlfFile):
        print('Input mlf file is invalid.')

    keys = adaptScp(args.inputScpFile, args.outputScpFile)
    convertMlf(args.inputMlfFile, args.inputPhoneListFile, args.outputLabelFile, keys)
