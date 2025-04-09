import os
import sys
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

  losses = {k: instantiate(v) for k, v in cfg['loss'].items()}
  print(f'Scheduler: {scheduler}')
  print(f'Optimizer: {optimizer}')
  print(f'Dataset path: {dataset_path}')
  print(f'Checkpoint dir: {checkpoint_dir}')
  print(f'Batch size: {batch_size}')
  print(f'Model: {model}')
  print(f'Losses: {losses}')


if __name__ == '__main__':
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
  print(sys.path)
  train()
