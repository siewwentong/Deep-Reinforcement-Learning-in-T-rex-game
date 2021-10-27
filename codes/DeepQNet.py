import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQNet(nn.Module):
    def __init__(self):
        super(DeepQNet, self).__init__()

        self.conv_layers = nn.Sequential(nn.Conv2d(4, 32, 5, stride=2, padding=0),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(),
                                         nn.Conv2d(32, 64, 5, stride=2, padding=0),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         nn.Conv2d(128, 128, 3, stride=1, padding=0),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU()
                                         )
        self.fc1 = nn.Linear(7 * 7 * 128, 512)
        self.fc2 = nn.Linear(512, 3)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # print(m)
                # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#
#
# def train_model(model, old_state, action, reward, new_state, game_over, optimiser, loss_func, device):
#     gamma = 0.9
#     # convert data to tensors
#     old_state = torch.tensor(old_state, dtype=torch.float)
#     action = torch.tensor(action, dtype=torch.long)
#     reward = torch.tensor(reward, dtype=torch.float)
#     new_state = torch.tensor(new_state, dtype=torch.float)
#     # reshape tensors to correct shape (for short term learning
#     if len(old_state.shape) == 3:
#         old_state = torch.unsqueeze(old_state, 0)
#         action = torch.unsqueeze(action, 0)
#         reward = torch.unsqueeze(reward, 0)
#         new_state = torch.unsqueeze(new_state, 0)
#         game_over = (game_over,)
#
#     old_state = old_state.to(device)
#     action = action.to(device)
#     reward = reward.to(device)
#     new_state = new_state.to(device)
#
#     model.train()
#     prediction = model(old_state)
#     target = prediction.clone()
#     for idx, gameover in enumerate(game_over):
#         if gameover:
#             Q = reward[idx]
#         else:
#             Q = reward[idx] + gamma * torch.max(model(torch.unsqueeze(new_state[idx], 0)))
#         action_idx = torch.argmax(action[idx]).item()
#         target[idx][action_idx] = Q
#         # f.write("Index:" + str(idx) + "\n")
#         # f.write("prediction: " + str(prediction[idx]) + "\n")
#         # f.write("target: " + str(target[idx]) + "\n")
#     # calculate loss
#     loss = loss_func(prediction, target)
#     # zero gradient and propagate gradient
#     optimiser.zero_grad()
#     loss.backward()
#     # learn
#     optimiser.step()
#     return loss.item() / len(action)

# import torch.nn as nn
#
#
# class DeepQNet(nn.Module):
#     def __init__(self):
#         super(DeepQNet, self).__init__()
#
#         self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True))
#         self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True))
#         self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True))
#
#         self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace=True))
#         self.fc2 = nn.Linear(512, 3)
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
#                 nn.init.uniform_(m.weight, -0.01, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, input):
#         output = self.conv1(input)
#         output = self.conv2(output)
#         output = self.conv3(output)
#         output = output.view(output.size(0), -1)
#         output = self.fc1(output)
#         output = self.fc2(output)
#
#         return output