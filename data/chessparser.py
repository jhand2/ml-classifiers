"""
Jordan Hand, Josh Malters, and Kevin Fong
CSE 415 Spring 2016
Final Project
Professor: S. Tanimoto

Parsing utility to parse text files into indivudal records for each file.
"""
import os


def parse_directory(dirname, label):
    """
    Parses an entire directory of text files, making a list entry for each
    file. For use if everything in a directory is the same class.

    Params:
        dirname (str): Full path to a directory
        label (str): The label of all the documents in directory dirname
    """
    records = []
    d = dirname.encode('unicode_escape')
    for f in os.listdir(d):
        path = os.path.join(d, f)
        if os.path.isfile(path):
            records.append(parse_file(f, path, label))
    return records


def parse_file(fname, path, label):
    """
    Parses a single file f into a record with text and a set of words.

    Params:
        f (str): Full path to a file
        label (str): The label of the document in file f
    """
    record = {'class': label}
    f = open(path, 'r')
    record['attribute'] = extract_words(f)
    record['name'] = fname.decode("utf-8")
    return record


def extract_words(f):
    """
    Takes a string f board and parses it into a list
    """
    board = []
    for line in f:
        temp = line.split(',')[:-1]
        board = []
        left = temp[::2]
        right = temp[1::2]
        piece = ['WKing:', 'WRook:', 'BKing:']
        for x in range(len(left)):
            board.append(piece[x] + left[x] + right[x])
    return frozenset(board)
