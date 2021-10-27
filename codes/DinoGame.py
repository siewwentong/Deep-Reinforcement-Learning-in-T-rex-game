import pygame
import os
import random
import cv2
import numpy as np
import torch

pygame.init()

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('test.avi', fourcc, 30, (SCREEN_WIDTH, SCREEN_HEIGHT))

# save_dir = "/content/gdrive/My Drive/units material/ECE4179/ChromeDinosaur-master/Assets" # for google colab
save_dir = os.path.join(os.getcwd(), "Assets")
RUNNING = [pygame.image.load(os.path.join(save_dir, "Dino/DinoRun1.png")),
           pygame.image.load(os.path.join(save_dir, "Dino/DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join(save_dir, "Dino/DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join(save_dir, "Dino/DinoDuck1.png")),
           pygame.image.load(os.path.join(save_dir, "Dino/DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join(save_dir, "Cactus/SmallCactus1.png")),
                pygame.image.load(os.path.join(save_dir, "Cactus/SmallCactus2.png")),
                pygame.image.load(os.path.join(save_dir, "Cactus/SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join(save_dir, "Cactus/LargeCactus1.png")),
                pygame.image.load(os.path.join(save_dir, "Cactus/LargeCactus2.png")),
                pygame.image.load(os.path.join(save_dir, "Cactus/LargeCactus3.png"))]

BIRD = [pygame.image.load(os.path.join(save_dir, "Bird/Bird1.png")),
        pygame.image.load(os.path.join(save_dir, "Bird/Bird2.png"))]

CLOUD = pygame.image.load(os.path.join(save_dir, "Other/Cloud.png"))

BG = pygame.image.load(os.path.join(save_dir, "Other/Track.png"))


class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        # changed this part of the code
        # 0 - run, 1 - jump, 2 - duck
        if userInput == 1 and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == 2 and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif not (self.dino_jump or userInput != 0):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8 # changed the velocity
        if self.dino_rect.y > self.Y_POS:
            self.dino_jump = False
            self.dino_rect.y = self.Y_POS
            self.jump_vel = self.JUMP_VEL

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self, game_speed):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH
        self.passed = False

    def update(self, game_speed, obstacles):
        remove_obstacles = False
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.remove(self)

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 300


class Bird(Obstacle):
    def __init__(self, image, height):
        self.type = 0
        super().__init__(image, self.type)
        # changed this part of the code
        self.rect.y = height
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index // 5], self.rect)
        self.index += 1


class DinoGameAI:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.reset_game()

    # changed this part of the code (added this function)
    def reset_game(self):
        self.SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.player = Dinosaur()
        self.cloud = Cloud()
        self.game_speed = 20
        self.points = 0
        self.obstacles = []

    def score(self):
        self.points += 1
        if self.points % 100 == 0:
            self.game_speed += 1

        # text = self.font.render("Points: " + str(self.points), True, (0, 0, 0))
        # textRect = text.get_rect()
        # textRect.center = (1000, 40)
        # self.SCREEN.blit(text, textRect)

    def background(self):
        image_width = BG.get_width()
        self.SCREEN.blit(BG, (self.x_pos_bg, self.y_pos_bg))
        self.SCREEN.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
        if self.x_pos_bg <= -image_width:
            self.SCREEN.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
            self.x_pos_bg = 0
        self.x_pos_bg -= self.game_speed

    # changed this part of the code (added this function)
    def run_game(self, action):
        game_over = False
        self.SCREEN.fill((255, 255, 255))

        self.player.draw(self.SCREEN)
        self.player.update(action)

        add_obstacles = False
        if len(self.obstacles) == 0:
            add_obstacles = True
        elif len(self.obstacles) == 1:
            min_distance = min(SCREEN_WIDTH // 3 + self.game_speed * 10, SCREEN_WIDTH)
            if self.obstacles[-1].rect.x < (SCREEN_WIDTH - random.randint(min_distance, SCREEN_WIDTH + 2)):
                add_obstacles = True

        if add_obstacles:
            if random.randint(0, 2) == 0:
                self.obstacles.insert(0, SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                self.obstacles.insert(0, LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 2) == 2:
                bird_height = [200, 245, 320]
                height = bird_height[random.randint(0, 2)]
                self.obstacles.insert(0, Bird(BIRD, height))

        reward = 0.1
        if action == 0:
            reward += 0.02
        for obstacle in self.obstacles:
            obstacle.draw(self.SCREEN)
            obstacle.update(self.game_speed, self.obstacles)
            if self.player.dino_rect.colliderect(obstacle.rect):
                # pygame.time.delay(2000)
                game_over = True
                reward = -1.0
            # if Dino cross obstacle
            elif ((obstacle.rect.x + obstacle.rect.width) < self.player.dino_rect.x) and not obstacle.passed:
                obstacle.passed = True
                reward = 1.0

        self.background()

        self.cloud.draw(self.SCREEN)
        self.cloud.update(self.game_speed)

        self.score()

        speed = 30
        self.clock.tick(speed)
        pygame.display.update()
        return game_over, self.points, reward

    # changed this part of the code (added this function)
    def screenshot(self, save_vid, out=None):
        # getting screenshot of game and convert into 3d array
        game_screenshot = pygame.display.get_surface()
        image = pygame.surfarray.array3d(game_screenshot)
        # fixing the orientation of the image
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.flip(image, 1)  # flip image horizontally
        if save_vid:
            out.write(image)
        # convert image to black and white
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return image


# if __name__ == "__main__":
#     game = DinoGameAI()
#     while True:
#         game.run_game(1)
#         # image = game.screenshot()
#         # cv2.imshow("image", image)
#         # cv2.waitKey(0)
