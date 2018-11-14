import os
import sys
import pygame
import random
from pygame import *

from src.utils.images import load_image, load_sprite_sheet
from src.utils.numeric import extractDigits
from src.utils.sounds import *

class Cloud(pygame.sprite.Sprite):
    def __init__(self, screen, x,y):
        pygame.sprite.Sprite.__init__(self,self.containers)
        self.screen = screen
        self.image,self.rect = load_image('cloud.png',int(90*30/42),30,-1)
        self.speed = 1
        self.rect.left = x
        self.rect.top = y
        self.movement = [-1*self.speed,0]

    def draw(self):
        self.screen.blit(self.image,self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()