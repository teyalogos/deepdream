"""
Runs while control_deepdream_show.py is running.
This script displays and loops over the images in /output

The script moves all of the images from /temp (which is where 
control_deepdream_show.py saves the deep dreams to) once it's 
reached a certain amount of files inside it.

"""

import pygame
from pygame import *
import pygame.image
import pygame.camera

import time
import os
import sys
import shutil

WIDTH = 1920
HEIGHT = 1080
windowSurface = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.camera.init()
cam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
cam.start()

def take_image(name):
    """Takes an image from the webcam"""
    img = cam.get_image()
    pygame.image.save(img, name)
    return img

def display_image(image_num):
    """Displays an image"""
    name = "output/"+str(image_num)+".jpg"
    img = pygame.image.load(name)
    img = pygame.transform.scale(img, (WIDTH, HEIGHT))
    
    windowSurface.blit(img, (0, 0)) #Replace (0, 0) with desired coordinates
    pygame.display.flip()

# take images to deep dream
#take_image("photo.jpg")

image_num = 0
limit = 200

while True:    

    # handle deep dreaming
    files = os.listdir("temp")
    temp_count = len(files)
    if len(files) >= limit:
        #take_image("photo.jpg")
        for f in files:
            os.remove("output/"+f)
            shutil.move("temp/"+f, "output")

    # handle image display
    display_image(image_num)
    if image_num >= (limit - 1):
        image_num = 1
    else:
        image_num += 1

    # handle fullscreen
    for e in event.get():
        if (e.type is KEYDOWN and e.key == K_f):
            if windowSurface.get_flags() & FULLSCREEN:
                pygame.display.set_mode((WIDTH, HEIGHT))
            else:
                pygame.display.set_mode((WIDTH, HEIGHT), FULLSCREEN)

    time.sleep(0.05)
