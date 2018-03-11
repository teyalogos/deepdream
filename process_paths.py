"""
process_paths.py
Usage: process_paths.py [directory]

1. Moves all files in specified directory to their own folder.
2. Appends all new paths to train.txt

This is used for one type of deep dream training method
"""
from sys import argv
import shutil
import os

_, dir = argv

file_num = 0
files = []


# clear files first
open("train.txt","w").close()
open("val.txt","w").close()

# write to file
with open("train.txt", "a") as train, open("val.txt", "a") as val:
    train.truncate()
    for f in os.listdir(dir):
        # store old file directory to move
        old_file = "{}/{}/{}".format(os.getcwd(), dir, f)
        # create new directory path to create and move file to
        new_dir = "{}/{}/{}".format(os.getcwd(), dir, str(file_num))

        # create new directory "new_dir" for f and move it there
        os.makedirs(new_dir)
        shutil.move(old_file, new_dir)

        # append new file name to train.txt and val.txt with it's number of image cateogry
        train.write("{}/{}/{} {}\n".format(dir, str(file_num), f, str(file_num)))
        val.write("{}/{}/{} {}\n".format(dir, str(file_num), f, str(file_num)))

        file_num += 1
