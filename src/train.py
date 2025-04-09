import os
import sys
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig


@hydra.main(config_path='../configs', config_name='config')
def train(cfg: DictConfig):
  # We simply print the configuration
  # print(OmegaConf.to_yaml(cfg))
  model = instantiate(cfg.model)

  print(cfg['env'])
  print(cfg['loss'])
  print(cfg['model'])
  print(cfg['optim'])
  print(cfg['sched'])

  scheduler = instantiate(cfg['sched'])
  optimizer = instantiate(cfg['optim'])
  dataset_path = cfg['env']['paths']['dataset']
  checkpoint_dir = cfg['env']['paths']['checkpoint']
  batch_size = cfg['batch_size']
  epochs = cfg['epochs']

  losses = [instantiate(v) for k, v in cfg['loss'].items() if k != 'weights']
  loss_names = cfg['loss']['weights'].keys()
  losses_dict = dict(zip(loss_names, tuple(zip(losses[0], cfg['loss']['weights'].values()))))
  weights_dict = cfg['loss']['weights']

  print(f'Scheduler: {scheduler}')
  print(f'Optimizer: {optimizer}')
  print(f'Dataset path: {dataset_path}')
  print(f'Checkpoint dir: {checkpoint_dir}')
  print(f'Batch size: {batch_size}')
  print(f'Model: {model}')
  print(f'Epochs: {epochs}')

  print(f'losses: {losses}')
  print(f'loss_names: {loss_names}')
  print(f'losses_dict: {losses_dict}')
  print(f'weights: {weights_dict}')

  # iou_loss = losses['loss']
  prediction = torch.randn((1, 1, 30, 30), dtype=torch.float32).sigmoid()
  target = torch.ones((1, 1, 30, 30), dtype=torch.float32)
  # print(f'iou_loss: {iou_loss(prediction, target)}')


if __name__ == '__main__':
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
  print(sys.path)
  train()
