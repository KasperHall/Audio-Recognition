from numpy.lib.function_base import append
import pygame
import csv
import os
import numpy as np
import speech_recognition as sr
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


pygame.init()

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = int(SCREEN_WIDTH*0.8)

screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
pygame.display.set_caption("Maze")

#Set framerate
clock = pygame.time.Clock()
FPS = 60

#Define player action variables
moving_left = False
moving_right = False
moving_up = False
moving_down = False

#Define map
BG = (0, 0, 20)

#Winner
is_winner = False
font = pygame.font.Font('freesansbold.ttf',52)
text_x = 400
text_y = 400

#Load model
model = keras.models.load_model('path/to/location')
commands = ['down', 'left', 'right', 'up']


def draw_bg():
    screen.fill(BG)


def callback(recognizer, audio, player):                          # this is called from the background thread
    try:
         wav = audio.get_segment(start_ms=200, end_ms=1200).get_wav_data()
         wave = decode_audio(wav)
         spectro = get_spectrogram(wave)
         spectro = tf.expand_dims(tf.expand_dims(spectro,-1), 0)
         prediction = model(spectro)
         prediction = commands[tf.math.argmax(prediction[0])]
         #prediction = recognizer.recognize_google(audio) # received audio data, now need to recognize it
        # if prediction == "down":
         #    player.moving_down = True
        # if prediction == "up":
        #     player.moving_up = True
        # if prediction == "left":
        #     player.moving_left = True
        # if prediction == "right":
        #     player.moving_right = True 
        
    except LookupError:
        print("Oops! Didn't catch that")
    except sr.UnknownValueError:
        print("Oops! Didn't catch that")

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_spectrogram(waveform):
  # Padding for files with less than 16000 samples
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=200, frame_step=400, fft_length=256)
      
  spectrogram = tf.math.pow(spectrogram, 0.2)
  spectrogram = tf.abs(spectrogram)

  return spectrogram

class Wall(pygame.sprite.Sprite):
    def __init__(self,x,y,scale,movable):
        self.movable = movable
        img = pygame.image.load(f'Walls/{self.movable}.png')
        img = pygame.transform.scale(img, (int(img.get_width() * scale), int(img.get_height() * scale)))
        self.image = img
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
    
    def draw(self):
        screen.blit(self.image, self.rect)


class Block(pygame.sprite.Sprite):
    def __init__(self, char_type, x, y, scale, speed):
        pygame.sprite.Sprite.__init__(self)
        self.char_type = char_type
        self.speed = speed
        self.direction = 1
        self.flip = False
        self.animation_list = []
        self.frame_index = 0
        self.action = 0
        self.moving_left = False
        self.moving_right = False
        self.moving_up = False
        self.moving_down = False
        self.update_time = pygame.time.get_ticks()
        self.velocity = 0
        temp_list = []
        for i in range(3):
            img = pygame.image.load(f'{self.char_type}/idle/{i}.png')
            img = pygame.transform.scale(img, (int(img.get_width() * scale), int(img.get_height() * scale)))
            temp_list.append(img)
        self.animation_list.append(temp_list)
        temp_list = []
        for i in range(6):
            img = pygame.image.load(f'{self.char_type}/running/{i+1}.png')
            img = pygame.transform.scale(img, (int(img.get_width()*scale), int(img.get_height()*scale)))
            temp_list.append(img)
        self.animation_list.append(temp_list)
        self.image = self.animation_list[self.action][self.frame_index]       
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
    
    def move(self, moving_left, moving_right, moving_up, moving_down, walls_rect, walls):
        # reset movement variables
        dx = 0
        dy = 0
        
        #assign movement variables if moving left or right
        if self.moving_left:
            dx = -self.speed
            self.flip = True
            self.direction = -1
        if self.moving_right:
            dx = self.speed
            self.flip = False
            self.direction = 1
        if self.moving_up:
            dy = -self.speed
        if self.moving_down:
            dy = self.speed
        
        collision_index = self.rect.collidelist(walls_rect)
        if collision_index != -1:
            wall = walls_rect[collision_index]
            if walls[collision_index].movable:
                return True

            if self.rect.top < wall.bottom and self.rect.bottom > wall.bottom and np.abs(self.rect.top - wall.bottom) < 5:#(self.rect.right > wall.left or self.rect.left < wall.right):
                self.rect.y = wall.bottom + 1
                self.moving_up = False
                self.velocity = 0


            elif self.rect.bottom > wall.top and np.abs(self.rect.bottom - wall.top) < 5:
                self.rect.y = wall.top - 48
                self.moving_down = False
                self.velocity = 0

            elif self.rect.right > wall.left and self.rect.left < wall.right and np.abs(self.rect.right - wall.left) < 5:
                self.rect.x = wall.left - 48
                self.moving_right = False
                self.velocity = 0

            else:
                self.rect.x = wall.right + 1
                self.moving_left = False
                self.velocity = 0

        
        else:
            self.rect.x += dx
            self.rect.y += dy
            
        
        #Update rectangle posision
        

    def update_animation(self):
        #update animation
        ANIMATION_COOLDOWN = 200
        #update image depending on current frame
        self.image = self.animation_list[self.action][self.frame_index]
        #check if enough time has passes since last update
        if pygame.time.get_ticks() - self.update_time > ANIMATION_COOLDOWN:
            self.update_time = pygame.time.get_ticks()
            self.frame_index += 1
        # if the animation has run out the reset to start
        if self.frame_index >= len(self.animation_list[self.action]):
            self.frame_index = 0
            
    def update_action(self, new_action):
        #check if the new action is different from the current one
        if new_action != self.action:
            self.action = new_action
            #update animation settings
            self.frame_index = 0
            self.update_time = pygame.time.get_ticks()

    def draw(self):
        screen.blit(pygame.transform.flip(self.image, self.flip, False), self.rect)




player = Block('Player', 80, 80, 2, 4)
walls_rect = []
walls = []
counter_x = 23
counter_y = 23
for j in range(2):
    for i in range(30):
        if not ((i == 28 or i == 27)and j==0):
            wall = Wall(counter_x, counter_y, 0.6, 0)
            counter_x += 40
            walls.append(wall)
            walls_rect.append(wall.rect)
        else:
            wall = Wall(counter_x, counter_y, 0.6, 1)
            counter_x += 40
            walls.append(wall)
            walls_rect.append(wall.rect)
    counter_y += 40*23
    counter_x = 23

counter_x = 23
counter_y = 63 
for j in range(2):
    for i in range(22):
        wall = Wall(counter_x, counter_y, 0.6, 0)
        counter_y += 40
        walls.append(wall)
        walls_rect.append(wall.rect)
    counter_x = 40*29.6
    counter_y = 63
x = 143
y = 63
for i in range(4):
    wall = Wall(x, y, 0.6, 0)
    walls.append(wall)
    walls_rect.append(wall.rect)
    y += 40

x = 63
y = 15*40+23
for i in range(5):
    wall = Wall(x,y,0.6,0)
    walls.append(wall)
    walls_rect.append(wall.rect)
    x += 40

x = 63
y = 18*40+23
for i in range(5):
    wall = Wall(x,y,0.6,0)
    walls.append(wall)
    walls_rect.append(wall.rect)
    x += 40

x = 12*40+23
y = 63
for i in range(23):
    if i != 7 and i != 8 and i != 9 and i != 10 and i != 11 and i != 15 and i != 16 and i != 17 and i != 18 and i != 19:
        wall = Wall(x,y,0.6,0)
        walls.append(wall)
        walls_rect.append(wall.rect)
        y += 40
    else:
        y += 40

x = 18*40+23
y = 17*40+23
for i in range(4):
    wall = Wall(x,y,0.6,0)
    walls.append(wall)
    walls_rect.append(wall.rect)
    y += 40

x = 17*40+23
y = 7*40+23
for i in range(2):
    wall = Wall(x,y,0.6,0)
    walls.append(wall)
    walls_rect.append(wall.rect)
    x += 40

x = 25*40+23
y = 10*40+23
for i in range(4):
    wall = Wall(x,y,0.6,0)
    walls.append(wall)
    walls_rect.append(wall.rect)
    x += 40

x = 22*40+23
y = 23
for i in range(4):
    wall = Wall(x,y,0.6,0)
    walls.append(wall)
    walls_rect.append(wall.rect)
    y += 40


run = True
r = sr.Recognizer()
m = sr.Microphone(sample_rate=16000)
with m as source:
  # we only need to calibrate once, before we start listening
  r.adjust_for_ambient_noise(source, duration=0.1)  

r.listen_in_background(m, lambda recognizer, audio: callback(recognizer, audio, player))
while run: 
    clock.tick(FPS)
    draw_bg()
    player.update_animation()
    player.draw()
    for w in walls:
        w.draw()


    #update player actions
    if player.moving_left or player.moving_right or player.moving_up or player.moving_down:
        player.update_action(1) # 1: run
    else:
        player.update_action(0) # 0: idle
    is_winner = player.move(moving_left, moving_right, moving_up, moving_down, walls_rect, walls)


    if is_winner:
        winner = font.render("Maze Completed", True, (255,255,255))
        screen.blit(winner,(text_x, text_y))
        player.moving_up = False
        


    for event in pygame.event.get():
        #quit game
        if event.type == pygame.QUIT:
            run = False


    pygame.display.update()


pygame.quit()