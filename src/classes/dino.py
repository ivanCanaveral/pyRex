import os
import sys
import pygame
import random
import pygame
from pygame import *

from src.utils.images import load_image, load_sprite_sheet
from src.utils.numeric import extractDigits
from src.utils.sounds import *

class Dino():
    def __init__(self, screen, sizex=-1,sizey=-1, scr_size=(600,150)):
        self.screen = screen
        self.scr_width = scr_size[0]
        self.scr_height = scr_size[1]
        self.gravity = 0.6
        self.images,self.rect = load_sprite_sheet('dino.png',5,1,sizex,sizey,-1)
        self.images1,self.rect1 = load_sprite_sheet('dino_ducking.png',2,1,59,sizey,-1)
        self.rect.bottom = int(0.98*self.scr_height)
        self.rect.left = self.scr_width/15
        self.image = self.images[0]
        self.index = 0
        self.counter = 0
        self.score = 0
        self.isJumping = False
        self.isDead = False
        self.isDucking = False
        self.isBlinking = False
        self.movement = [0,0]
        self.jumpSpeed = 11.5
        self.last_instruction = 0
        self.JUMP_ORDER = 2
        self.DUCK_ORDER = 1

        self.stand_pos_width = self.rect.width
        self.duck_pos_width = self.rect1.width

    def draw(self):
        self.screen.blit(self.image,self.rect)

    def checkbounds(self):
        if self.rect.bottom > int(0.98*self.scr_height):
            self.rect.bottom = int(0.98*self.scr_height)
            self.isJumping = False

    def jump(self):
        if self.rect.bottom == int(0.98 * self.scr_height):
            self.isJumping = True
            if pygame.mixer.get_init() != None:
                jump_sound.play()
            self.movement[1] = - self.jumpSpeed

    def duck(self):
        if not (self.isJumping and self.isDead):
            self.isDucking = True

    def stand_up(self):
        self.isDucking = False

    def autopilot(self, action):
        if action != self.last_instruction:
            if action == self.JUMP_ORDER:
                self.jump()
            elif action == self.DUCK_ORDER:
                self.duck()
            else:
                self.stand_up()
        self.last_instruction = action
        
    def check_collision(self, obj):
        if pygame.sprite.collide_mask(self, obj):
            self.isDead = True
            if pygame.mixer.get_init() != None:
                die_sound.play()

    def update(self):
        if self.isJumping:
            self.movement[1] = self.movement[1] + self.gravity

        if self.isJumping:
            self.index = 0
        elif self.isBlinking:
            if self.index == 0:
                if self.counter % 400 == 399:
                    self.index = (self.index + 1)%2
            else:
                if self.counter % 20 == 19:
                    self.index = (self.index + 1)%2

        elif self.isDucking:
            if self.counter % 5 == 0:
                self.index = (self.index + 1)%2
        else:
            if self.counter % 5 == 0:
                self.index = (self.index + 1)%2 + 2

        if self.isDead:
           self.index = 4

        if not self.isDucking:
            self.image = self.images[self.index]
            self.rect.width = self.stand_pos_width
        else:
            self.image = self.images1[(self.index)%2]
            self.rect.width = self.duck_pos_width

        self.rect = self.rect.move(self.movement)
        self.checkbounds()

        if not self.isDead and self.counter % 7 == 6 and self.isBlinking == False:
            self.score += 1
            if self.score % 100 == 0 and self.score != 0:
                if pygame.mixer.get_init() != None:
                    checkPoint_sound.play()

        self.counter = (self.counter + 1)