"""
process_images.py
Usage: python process_images.py [dir] [width] [height]

1. Ignores non jpegs
2. Ignores corrupt jpegs
3. Resizes all images to 256x256
4. Returns the image mean and number of images

"""

from PIL import Image
from sys import argv
import os 

_, dir, width, height = argv

width = int(width)
height = int(height)

file_num = 0
original_file_num = 0

# iterate over all files in directory
for f in os.listdir(dir):
    try:
        # open image and notify user
        img = Image.open("{}{}".format(dir, f))
        print("Resizing {}: {}{}".format(file_num, dir, f))

        # resize and save image
        img = img.resize((width, height), Image.ANTIALIAS)
        img.save("resized/{}{}".format(str(file_num), '.jpg'))

        file_num += 1
    except FileNotFoundError:
        os.makedirs("resized")
    except OSError:q
        print("File {}{} is bad, not resizing".format(dir, f))
        pass

    original_file_num += 1

print("Number of images processed: {}\nOriginal number of images; {} ".format(file_num, original_file_num))
os.system("convert {}/../{}-resized/*.jpg -average res.png; identify -verbose res.png".format(dir, dir))