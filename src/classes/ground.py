import os
import sys
import pygame
import random
from pygame import *

from src.utils.images import load_image, load_sprite_sheet
from src.utils.numeric import extractDigits
from src.utils.sounds import *

class Ground():
    def __init__(self, screen, speed=-5, scr_size=(600,150)):
        self.screen = screen
        self.scr_width = scr_size[0]
        self.scr_height = scr_size[1]
        self.image,self.rect = load_image('ground.png',-1,-1,-1)
        self.image1,self.rect1 = load_image('ground.png',-1,-1,-1)
        self.rect.bottom = self.scr_height
        self.rect1.bottom = self.scr_height
        self.rect1.left = self.rect.right
        self.speed = speed

    def draw(self):
        self.screen.blit(self.image,self.rect)
        self.screen.blit(self.image1,self.rect1)

    def update(self):
        self.rect.left += self.speed
        self.rect1.left += self.speed

        if self.rect.right < 0:
            self.rect.left = self.rect1.right

        if self.rect1.right < 0:
            self.rect1.left = self.rect.right