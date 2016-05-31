Machine Learning Classifiers
============================

This project implements various machine learning topics/algorithms and attempts
to provde some transparency into how they are working and what kind of accuracy
these algorithms achieve.

Currently Available Data Sets
-----------------------------

See usage for how to run these data sets with classify.py

fruit:

    Contains fruits which are classified as either lemon, banana, or other.

    This is the smallest data set. It is more of a toy data set to show how
    the program works

emails:

    This data set contains emails released from enron and classified as either
    spam or "ham" (aka not spam).

    This data set takes some time with k nearest neighbors and bagging (40-50
    second). It is probably the most interesting to play with and sees the best
    accuracy from our classifiers.

chess:

    This data set has various records of the location of each king and the
    White rook in a chess game. The boards are classified by how many moves
    it takes for either player to win the game.

    This is the most experimental of our sets. It takes the longest for our
    classifiers to classify (this is due to its size) and is the least
    accurately classified by all of our classifiers (this is likely due to the
    complexity of chess and the limited scope of each data point).

Usage:
------

Run the classify.py file with python 3.5 to interact with the program. Pass as
an argument the data model you wish to use. Available data models are

chess
fruit
emails

Example:

python3.5 classify.py emails

More info:
----------

More information can be found in report.txt

Code style guidelines can be found in style_guide.txt

Directories contain the following:

    data: Actual data records as well as python files to model these data
          sets and allow operations on them.

    models: Python files for each of the actual classifiers

    metrics: Python files for running tests and evaluation metrics on
             classifiers
