import os
import json
import time
import math
from collections import defaultdict, OrderedDict
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid

from habitat import logger
from habitat.config.default import Config
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import append_text_to_image
from baselines.common.viz_utils import (
    observations_to_image
)

from habitat_baselines.utils.common import generate_video
from baselines.common.environments import MultiObjNavRLEnv
from baselines.common.utils import extract_scalars_from_info
from multion import maps as multion_maps
from semantic_segmentation.rednet import RedNet
from semantic_segmentation.multion_semantic_data import *
from semantic_segmentation.util import CrossEntropyLoss2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OBJECT_MAP = {0: 'cylinder_red', 1: 'cylinder_green', 2: 'cylinder_blue', 3: 'cylinder_yellow', 
              4: 'cylinder_white', 5:'cylinder_pink', 6: 'cylinder_black', 7: 'cylinder_cyan'}
METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "raw_metrics", "traj_metrics"}

# -- Create RedNet model
cfg_rednet = {
    'arch': 'rednet',
    'resnet_pretrained': False,
    'finetune': True,
    'SUNRGBD_pretrained_weights': '',
    'n_classes': 9,
    'upsample_prediction': True,
    'load_model': 'data/sem_seg/rednet_mp3d_best_model.pkl',
}

def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return step, epoch, model
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)

def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch, local_count, num_train):
    
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # usually this happens only on the start of a epoch
    epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch_float)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))
    
def print_log(global_step, epoch, local_count, count_inter, dataset_size, loss, time_inter):
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} ({:3.1f}%)]    '
          'Loss: {:.6f} [{:.2f}s every {:>4} data]'.format(
        global_step, epoch, local_count, dataset_size,
        100. * local_count / dataset_size, loss.data, time_inter, count_inter))
    
def color_label(label):
    label = label.clone().cpu().data.numpy()
    # colored_label = np.vectorize(lambda x: multion_maps.OBJECT_MAP_COLORS[x])
    # colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = multion_maps.OBJECT_MAP_COLORS[label].astype(np.float32)
    colored = colored.squeeze()

    try:
        return torch.from_numpy(colored.transpose([0, 3, 1, 2]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])
    
def convert_weights_cuda_cpu(weights, device):
    names = list(weights.keys())
    is_module = names[0].split('.')[0] == 'module'
    if device == 'cuda' and not is_module:
        new_weights = {'module.'+k:v for k,v in weights.items()}
    elif device == 'cpu' and is_module:
        new_weights = {'.'.join(k.split('.')[1:]):v for k,v in weights.items()}
    else:
        new_weights = weights
    return new_weights

def train(config: Config):
    
    # image_h=256
    # image_w=256
    # train_data = MultionSemanticData(transform=transforms.Compose([scaleNorm(),
    #                                                                RandomScale((1.0, 1.4)),
    #                                                                RandomHSV((0.9, 1.1),
    #                                                                                      (0.9, 1.1),
    #                                                                                      (25, 25)),
    #                                                                RandomCrop(image_h, image_w),
    #                                                                RandomFlip(),
    #                                                                ToTensor(),
    #                                                                Normalize()]),
    #                                  phase_train=True,
    #                                  training_data=config.IL.RedNet.training_data)
    
    train_data = MultionSemanticData(phase_train=True,
                                     training_data=config.IL.RedNet.training_data)
    
    train_loader = DataLoader(train_data, batch_size=config.IL.RedNet.batch_size, shuffle=True,
                              num_workers=config.IL.RedNet.workers, pin_memory=False)

    num_train = len(train_data)

    model = RedNet(config.IL.RedNet)
    model.to(device)
    
    # print('Loading pre-trained weights: ', cfg_rednet['load_model'])
    # state = torch.load(cfg_rednet['load_model'])
    # model_state = state['model_state']
    # model_state = convert_weights_cuda_cpu(model_state, 'cpu')
    # model.load_state_dict(model_state)
    
    model.train()
        
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    if "loss_type" in config.IL.RedNet and config.IL.RedNet.loss_type == "cel":
        sem_loss = nn.CrossEntropyLoss()
    else:
        sem_loss = CrossEntropyLoss2d()
    
    sem_loss.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=float(config.IL.RedNet.lr),
                                momentum=config.IL.RedNet.momentum, weight_decay=float(config.IL.RedNet.weight_decay))

    global_step = 0
    start_epoch = config.IL.RedNet.start_epoch

    global_step, start_epoch, model = load_ckpt(model, optimizer, config.IL.RedNet.last_ckpt, device)

    lr_decay_lambda = lambda epoch: config.IL.RedNet.lr_decay_rate ** (epoch // config.IL.RedNet.lr_epoch_per_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

    writer = SummaryWriter(config.RESULTS_DIR)

    for epoch in range(int(start_epoch), config.IL.RedNet.epochs):

        scheduler.step(epoch)
        local_count = 0
        last_count = 0
        end_time = time.time()
        if epoch % config.IL.RedNet.save_epoch_freq == 0 and epoch != start_epoch:
            save_ckpt(config.CHECKPOINT_FOLDER, model, optimizer, global_step, epoch,
                      local_count, num_train)

        for batch_idx, sample in enumerate(train_loader):

            image = sample['image'].to(device).permute(0,3,1,2)
            depth = sample['depth'].to(device).unsqueeze(1)
            targets = sample['label'].type(torch.long).to(device)
            
            optimizer.zero_grad()
            preds, _, _, _, _ = model(image,depth)
            loss = sem_loss(preds, targets)
            loss.backward()
            optimizer.step()
            
            local_count += image.data.shape[0]
            global_step += 1
            
            if global_step % config.IL.RedNet.print_freq == 0 or global_step == 1:

                time_inter = time.time() - end_time
                count_inter = local_count - last_count
                print_log(global_step, epoch, local_count, count_inter,
                          num_train, loss, time_inter)
                end_time = time.time()

                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step, bins='doane')
                grid_image = make_grid(image.clone().cpu().data, 3, normalize=True)
                writer.add_image('image', grid_image, global_step)
                grid_image = make_grid(depth.clone().cpu().data, 3, normalize=True)
                writer.add_image('depth', grid_image, global_step)
                grid_image = make_grid(color_label(torch.max(preds, 1)[1]), 3, normalize=False,
                                       range=(0, 255))
                writer.add_image('Predicted label', grid_image, global_step)
                grid_image = make_grid(color_label(targets), 3, normalize=False,
                                       range=(0, 255))
                writer.add_image('Groundtruth label', grid_image, global_step)
                writer.add_scalar('CrossEntropyLoss', loss.data, global_step=global_step)
                writer.add_scalar('Learning rate', scheduler.get_lr()[0], global_step=global_step)
                last_count = local_count

    save_ckpt(config.CHECKPOINT_FOLDER, model, optimizer, global_step, config.IL.RedNet.epochs,
              0, num_train)

    print("Training completed ")

def evaluate_agent(config: Config) -> None:
    
    model_rednet = RedNet(config.IL.RedNet)
    model_rednet = model_rednet.to(device)
    global_step, start_epoch, model_rednet = load_ckpt(model_rednet, None, config.IL.RedNet.last_ckpt, device)
    model_rednet.eval()
    
    split = config.EVAL.SPLIT
    config.defrost()
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.DATASET.SPLIT = split
    if len(config.VIDEO_OPTION) > 0:
        config.defrost()
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
    config.freeze()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    logger.add_filehandler(config.LOG_FILE)

    env = MultiObjNavRLEnv(config=config)

    agent = OracleAgent(config.TASK_CONFIG, env)

    stats = defaultdict(float)
    num_episodes = (min(config.TEST_EPISODE_COUNT, len(env.episodes)) 
                    if config.TEST_EPISODE_COUNT > 0 
                    else len(env.episodes))
    
    for i in tqdm(range(num_episodes)):
        obs = env.reset()
        agent.reset()
        done = False

        rgb_frames = []
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            output, _ = model_rednet(torch.FloatTensor(obs["rgb"]).permute(2,0,1).unsqueeze(0).to(device), 
                                     torch.FloatTensor(obs["depth"]).permute(2,0,1).unsqueeze(0).to(device))
            #output = output.detach().cpu().numpy().astype(np.uint8)
            output = torch.max(output, 1)[1]
            
            if len(config.VIDEO_OPTION) > 0:
                #output_viz = color_label(torch.max(pred, 1)[1] + 1)[0]
                frame = observations_to_image(obs, pred_sem_obs=output[0], info=info, action=[action])
                
                rgb_frames.append(frame)
        
        if len(config.VIDEO_OPTION) > 0:
            generate_video(
                video_option=config.VIDEO_OPTION,
                video_dir=config.VIDEO_DIR,
                images=rgb_frames,
                episode_id=f"{os.path.basename(env.current_episode.scene_id)}_{env.current_episode.episode_id}",
                checkpoint_idx=0,
                metrics=extract_scalars_from_info(info, METRICS_BLACKLIST),
                tb_writer=None,
            )
            
        if "top_down_map" in info:
            del info["top_down_map"]
        if "collisions" in info:
            del info["collisions"]
            
        for m, v in info.items():
            stats[m] += v

    stats = {k: v / num_episodes for k, v in stats.items()}
    stats["num_episodes"] = num_episodes

    logger.info(f"Averaged benchmark for {num_episodes} episodes:")
    for stat_key in stats.keys():
        logger.info("{}: {:.3f}".format(stat_key, stats[stat_key]))

    with open(config.RESULTS_DIR + f"/stats_epi{num_episodes}_{split}.json", "w") as f:
        json.dump(stats, f, indent=4)

def collate_fn(batch):
    return tuple(zip(*batch))

class OracleAgent(Agent):
    def __init__(self, task_config: Config, env):
        self.actions = [
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]
        self.env = env._env

    def reset(self):
        self.follower = ShortestPathFollower(
            self.env.sim, goal_radius=0.25, return_one_hot=False,
            stop_on_error=True # False for debugging
        )
        self.num_tries = 10

    def act(self, observations):
        current_goal = self.env.task.current_goal_index
        
        best_action = 0
        try:
            best_action = self.follower.get_next_action(self.env.current_episode.goals[current_goal].position)
        except:
            while self.num_tries > 0:
                best_action = np.random.choice(self.actions)
                self.num_tries -= 1
                
        return best_action
    