import os
import sys
import pygame
import random
from pygame import *

from src.utils.images import load_image, load_sprite_sheet
from src.utils.numeric import extractDigits
from src.utils.sounds import *

class Cactus(pygame.sprite.Sprite):
    def __init__(self, screen, speed=5, sizex=-1, sizey=-1, scr_size=(600,150)):
        pygame.sprite.Sprite.__init__(self,self.containers)
        self.screen = screen
        self.scr_width = scr_size[0]
        self.scr_height = scr_size[1]
        self.images,self.rect = load_sprite_sheet('cacti-small.png',3,1,sizex,sizey,-1)
        self.rect.bottom = int(0.98*self.scr_height)
        self.rect.left = self.scr_width + self.rect.width
        self.image = self.images[random.randrange(0,3)]
        self.movement = [-1*speed,0]

    def draw(self):
        self.screen.blit(self.image,self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)

        if self.rect.right < 0:
            self.kill()