import os
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from DinoGame import DinoGameAI
from DeepQNet import DeepQNet, train_model
from collections import deque
import shelve
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output

# TODO
# check weights and gradient (see whether learning is happening)

INPUT_SIZE = 90
BATCH_SIZE = 100
OBSERVATION_STEP = 200
EXPLORATION_STEP = 1000
INITIAL_EPSILON = 0.9
FINAL_EPSILON = 1e-3


def stack_images(image, stacked_images):
    while len(stacked_images) < 3:
        stacked_images = np.append(stacked_images, image, axis=0)
    stacked_images = np.append(stacked_images, image, axis=0)
    if len(stacked_images) > 4:
        stacked_images = np.delete(stacked_images, 0, axis=0)
    return stacked_images


class Agent:
    def __init__(self, train, saved_dir, f):
        self.no_games = 0
        self.epsilon = INITIAL_EPSILON if train else 0  # to control the randomness during training
        self.saved_memory = shelve.open(os.path.join(saved_dir,'memory.pickle'), writeback=True)
        if 'memory' in self.saved_memory:
            self.memory = self.saved_memory['memory']
            if f:
                # print('getting from saved memory')
                f.write('getting from saved memory\n')
                # print('length: ', len(self.memory))
                f.write('length: ' + str(len(self.memory)) + '\n')
        else:
            if f:
                # print('starting from scratch')
                f.write('starting from scratch\n')
            self.memory = deque(maxlen=10000)
            self.saved_memory['memory'] = self.memory
        self.stacked_im = np.zeros([1, INPUT_SIZE, INPUT_SIZE])

    def get_state(self, game):
        save_vid = True if (self.no_games % 5 == 0) else False
        image = game.screenshot(save_vid)
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        self.stacked_im = stack_images(np.expand_dims(image, axis=0), self.stacked_im)
        # stacked_im = torch.from_numpy(self.stacked_im).float()
        return self.stacked_im

    def remember(self, old_state, action, reward, new_state, game_over):
        self.memory.append((old_state, action, reward, new_state, game_over))

    def train_long_term_memory(self, model, optimiser, loss_func, device):
        loss = None
        if len(self.memory) > OBSERVATION_STEP:  # if enough instances
            batch_data = random.sample(self.memory, BATCH_SIZE)
            old_state, action, reward, new_state, game_over = zip(*batch_data)
            loss = train_model(model, old_state, action, reward, new_state, game_over, optimiser, loss_func, device)
        return loss

    def get_action(self, model, state, device):
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EXPLORATION_STEP
        if random.random() > self.epsilon:
            state = torch.unsqueeze(torch.tensor(state).float(), 0)
            actions = model(state.to(device))
            # print(actions)
            # exit()
            action = torch.argmax(actions).item()
            # print(action)
        else:
            action = random.randint(0, 2)
        return action


def train(target, saved_dir, device):
    f = open("log.txt", "w")
    game = DinoGameAI()
    model = DeepQNet().to(device)
    agent = Agent(True, saved_dir, f)
    score = []
    loss_log = []
    optimiser = optim.SGD(model.parameters(), lr=1e-3)
    loss_func = nn.SmoothL1Loss()  # less sensitive to outliers compared to MSELoss
    highest_score = 0
    if os.path.isfile(os.path.join(saved_dir, "model.pt")):
        # load Checkpoint
        if f:
            # print("getting checkpoint from previous training")
            f.write("getting checkpoint from previous training")
        check_point = torch.load(os.path.join(saved_dir, "model.pt"))
        model.load_state_dict(check_point['model_state_dict'])
        optimiser.load_state_dict(check_point['optimizer_state_dict'])
        highest_score = check_point['highest_score']
        agent.no_games = check_point['no_games']
        score = check_point['score']
        loss_log = check_point['loss_log']
        agent.epsilon = check_point['epsilon']
    if f:
        f.write("Starting with: \n")
        f.write("Cumulative number of games: " + str(agent.no_games) + '\n')
        f.write("Highest score: " + str(highest_score))
        f.write("Epsilon: " + str(agent.epsilon))
    game.run_game(0)
    while highest_score < target:
        # get old state
        old_state = agent.get_state(game)
        # get action
        action = agent.get_action(model, old_state, device)
        # get new state and reward
        game_over, points, reward = game.run_game(action)
        new_state = agent.get_state(game)
        # train short term memory
        agent.remember(old_state, action, reward, new_state, game_over)
        # train long term memory
        loss = agent.train_long_term_memory(model, optimiser, loss_func, device)
        if loss is not None:
            loss_log.append(loss)

        if game_over:
            agent.no_games += 1
            game.reset_game()
            score.append(points)
            # plot score
            clear_output(True)
            f = plt.figure(figsize=(18, 5))
            f.add_subplot(1, 2, 1)
            plt.plot(score)
            plt.xlabel("No of games")
            plt.ylabel("Score")
            plt.title("Model Performance")
            f.add_subplot(1, 2, 2)
            plt.plot(loss_log)
            plt.xlabel("No of training steps")
            plt.ylabel("loss")
            plt.title("Model Loss")
            plt.show()
            plt.savefig(os.path.join(saved_dir,'plots.png'))
            if points >= highest_score:
                highest_score = points
                print("highest score", highest_score)
                # save model
                highest_score = points
                torch.save({
                    'model_state_dict': model.state_dict(),
                }, os.path.join(saved_dir, "best_model.pt"))
        if agent.no_games % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'highest_score': highest_score,
                'no_games': agent.no_games,
                'score': score,
                'loss_log': loss_log,
                'epsilon': agent.epsilon
            }, os.path.join(saved_dir, "model.pt"))
    game.vid.release()


def deploy(saved_dir, device):
    # f = open("test_log.txt", "w")
    model = DeepQNet().to(device)
    agent = Agent(False, saved_dir, None)
    game = DinoGameAI()
    # load Checkpoint
    check_point = torch.load(os.path.join(saved_dir, "model.pt"))
    model.load_state_dict(check_point['model_state_dict'])
    game_over = False
    while not game_over:
        # get old state
        state = agent.get_state(game)
        game.screenshot(True) # save video
        # get action
        action = agent.get_action(model, state, device)
        # get new state and reward
        game_over, points, _ = game.run_game(action)
    print("Total points: ", points)


if __name__ == "__main__":
    save_dir = os.getcwd()
    train(200, save_dir, 'cpu')
    # deploy()
