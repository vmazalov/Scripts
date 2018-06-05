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
                key = key.replace('/', '\\')
                result[key] = newKey
                outputLine = newKey + "=" + splitted[1]
                outputFile.write(outputLine)
                counter += 1
    return result

def adaptLattice(latticeFile, output, keys):
    counter = 0
    with open(output, "w") as outputFile:
        with open(latticeFile) as f:
            for line in f:
                splitted = line.split("=")
                if len(splitted) != 2:
                    raise Exception("Invalid input '%s', line with invalid format '%s'" % (latticeFile, line))
                key = splitted[0].strip()
                if key.endswith(".mfc"):
                    key = key[:-4]
                key = key.replace('/', '\\')
                if key in keys:
                    outputLine=keys[key] + "=" + splitted[1]
                    outputFile.write(outputLine)
                else:
                    outputFile.write(outputLine)
                    print("Unknown lattice key '%s'" % key)

def adaptMlf(mlfFile, output, keys):
    previous = ""
    with open(output, "w") as outputFile:
        with open(mlfFile) as inputFile:
            for originalLine in inputFile:
                line = originalLine.strip()
                if previous == "." or previous == "#!MLF!#":
                    line = line.strip('"')
                    if line.endswith(".lab"):
                        line = line[:-4]
                    line = line.replace('/', '\\')
                    if line in keys:
                        line = '"' + keys[line] + '"' + "\n"
                        outputFile.write(line)
                    else:
                        outputFile.write(originalLine)
                        print("Unknown MLF key '%s'" % line)
                else:
                    outputFile.write(originalLine)
                previous = line.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapts mlf and scp files to use logical keys of smaller length")
    parser.add_argument('--inputScpFile', help='Input scp file.', required=True)
    parser.add_argument('--inputMlfFile', help='Input mlf file that corresponds to the scp file.', required=True)
    parser.add_argument('--inputLatticeFile', help='Input lattice file.', required=False)
    parser.add_argument('--outputScpFile', help='Output scp file.', required=True)
    parser.add_argument('--outputMlfFile', help='Output mlf file.', required=True)
    parser.add_argument('--outputLatticeFile', help='Output mlf file.', required=False)
    args = parser.parse_args()
    
    if not os.path.exists(args.inputScpFile):
        print('Input scp file is invalid.')

    if not os.path.exists(args.inputMlfFile):
        print('Input mlf file is invalid.')

    if args.inputLatticeFile and not os.path.exists(args.inputLatticeFile):
        print('Input lattice file is invalid')

    keys = adaptScp(args.inputScpFile, args.outputScpFile)
    adaptMlf(args.inputMlfFile, args.outputMlfFile, keys)
    if args.inputLatticeFile:
        adaptLattice(args.inputLatticeFile, args.outputLatticeFile, keys)
