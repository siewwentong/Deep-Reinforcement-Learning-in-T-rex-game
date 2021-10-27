import Agent
import os
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target", help="target score (integer)", required=True)
args = parser.parse_args()

save_dir = os.getcwd()
os.environ["SDL_VIDEODRIVER"] = "dummy"
GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')
print("device = ", device)
Agent.train(int(args.target), save_dir, device)