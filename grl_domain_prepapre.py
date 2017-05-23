import os
import argparse

def adaptScp(scpFile):
    result = {}
    counter = 0
    with open(scpFile) as f:
        for line in f:
            splitted = line.split("=")
            #print(line)
            if len(splitted) != 2:
                print(counter)
                raise Exception("Invalid input '%s', line with invalid format '%s'" % (scpFile, line))
            key = splitted[0].strip()
            if key.endswith(".mfc"):
                key = key[:-4]
            newKey = str(counter)
            result[key] = newKey
            counter += 1
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts domain labels from key-value pair file, phone list file and the MLF file")
    parser.add_argument('--inputScpFile', help='Input scp file with txt uttrs IDs.', required=True)
    parser.add_argument('--outputDomainLabelFile', help='Input scp file.', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.inputScpFile):
        print('Input scp file is invalid.')

    keys = adaptScp(args.inputScpFile)

    with open(args.outputDomainLabelFile, "w") as outputFile:
        with open(args.inputScpFile) as f:
            for line in f:
                splitted = line.split("=")
                if len(splitted) != 2:
                    raise Exception("Invalid input '%s', line with invalid format '%s'" % (scpFile, line))
                key = splitted[0].strip()
                if key.endswith(".mfc"):
                    key = key[:-4]

                frameBound = splitted[1].split("[")[1].split("]")[0].split(",")
                if len(frameBound) != 2:
                    raise Exception("Invalid frame bound '%s', line with invalid format '%s'" % (scpFile, line))

                label = "1" if key.endswith("_n") else "0"
                k = keys[key]
                for i in range(int(frameBound[0]), int(frameBound[1]) + 1):
                   outputFile.write(keys[key] + " |l " + label + ":1\n")