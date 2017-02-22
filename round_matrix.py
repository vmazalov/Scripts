import numpy
import os
import argparse


parser = argparse.ArgumentParser(description="Read matrix from a text file, round its values and print to output file")
parser.add_argument('--inputFile', help='Input file.', required=True)
parser.add_argument('--skipLines', help='Number of lines to skip from the top of the file', required=False)
parser.add_argument('--outputFile', help='Output file.', required=True)
args = parser.parse_args()

if not os.path.exists(args.inputFile):
    print('Input file is invalid.')


if args.skipLines != "":
    skipLines = int(args.skipLines)
else:
    skipLines = 0

matrix = numpy.loadtxt(args.inputFile,skiprows=skipLines)
numpy.around(matrix, decimals=1)
numpy.savetxt(args.outputFile, matrix, fmt="%0.5f")
