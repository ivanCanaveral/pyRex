import os
import sys
import pygame
import random
from pygame import *

from src.utils.images import load_image, load_sprite_sheet
from src.utils.numeric import extractDigits
from src.utils.sounds import *

class Ptera(pygame.sprite.Sprite):
    def __init__(self, screen, speed=5, sizex=-1, sizey=-1, scr_size=(600,150)):
        pygame.sprite.Sprite.__init__(self,self.containers)
        self.screen = screen
        self.scr_width = scr_size[0]
        self.scr_height = scr_size[1]
        self.images,self.rect = load_sprite_sheet('ptera.png',2,1,sizex,sizey,-1)
        self.ptera_height = [self.scr_height*0.75,self.scr_height*0.75,self.scr_height*0.50]
        self.rect.centery = self.ptera_height[random.randrange(0,3)]
        self.rect.left = self.scr_width + self.rect.width
        self.image = self.images[0]
        self.movement = [-1*speed,0]
        self.index = 0
        self.counter = 0

    def draw(self):
        self.screen.blit(self.image,self.rect)

    def update(self):
        if self.counter % 10 == 0:
            self.index = (self.index+1)%2
        self.image = self.images[self.index]
        self.rect = self.rect.move(self.movement)
        self.counter = (self.counter + 1)
        if self.rect.right < 0:
            self.kill()