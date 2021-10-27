# from MONARCH
import torch
import matplotlib.pyplot as plt

checkpoint = torch.load("trained_models/chrome_dino.pth")
loss_log = checkpoint['loss_log']
score_log = checkpoint['score_log']
reward_log = checkpoint['reward_log']
action_log = checkpoint['action_log']

# plot graph
f = plt.figure(figsize=(40, 20))
f.add_subplot(2, 2, 1)
plt.plot(reward_log)
plt.xlabel("No of training steps")
plt.ylabel("reward")
plt.title("Reward Obtained for Action taken")
f.add_subplot(2, 2, 2)
plt.plot(loss_log)
plt.xlabel("No of training steps")
plt.ylabel("loss")
plt.title("Model Loss")
f.add_subplot(2, 2, 3)
plt.plot(action_log)
plt.xlabel("No of training steps")
plt.ylabel("action taken")
plt.title("Action taken by model")
f.add_subplot(2, 2, 4)
plt.plot(score_log)
plt.xlabel("No of games")
plt.ylabel("Score")
plt.title("Model Performance")
# plt.show()
plt.savefig('plots.png')