import cv2
import sys
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import time
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.unet import Generator
from src.dataset import TestDataset, BinarizeTransform
from src.utils import next_exp_folder
from src.config import CHECKPOINT_PATH

# fixing seeds during inference
def set_random_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

set_random_seed(42)

parser = argparse.ArgumentParser(description = 'Performing inference on topology images')
parser.add_argument('inference_folder', type = str, help = 'Relative path to an inference image folder')
parser.add_argument('--batch_size', type = int, default = 2, help = 'Batch size for inference')
args = parser.parse_args()

DATA_PATH = args.inference_folder # 'data/processed/gds_dataset/origin/test_origin'
MODEL_PATH = os.path.join(CHECKPOINT_PATH, 'exp_9/last_checkpoint.pth') #'/mnt/data/amoskovtsev/mb_opc/checkpoints/exp_3/last_checkpoint.pth'
OUTPUT_DIR = next_exp_folder('inference/output_img')
BATCH_SIZE = args.batch_size
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running inference on device: {device}')

def save_image(output_batch, checkpoint_dir="checkpoints", image_type='true_correction'):
    for i,output in enumerate(output_batch):
        single_image = output.squeeze(dim=0)
        single_image[single_image > 0.5] = 1.0
        single_image[single_image <= 0.5] = 0.0

        single_image_path = image_type[i].split('/')[-1][:-4]
        img_save_path = os.path.join(checkpoint_dir, f"{single_image_path}.jpg")
        cv2.imwrite(f"{img_save_path}", (single_image*255).detach().cpu().numpy())
        print(f"Saved generated image at {img_save_path}")

generator_model = Generator(in_ch = 1, out_ch = 1)
# generator_model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
generator_model.load_state_dict(torch.load(MODEL_PATH, map_location = device)['model_state_dict'])
generator_model = generator_model.to(device)
generator_model.eval()
print('Model initialized:', generator_model)
print(f'Total number of parametres in neural network:{sum(p.numel() for p in generator_model.parameters())}')

TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Grayscale(),
    BinarizeTransform(threshold=0.5)
])

# Dataset paths
TEST_DATASET = TestDataset(DATA_PATH, transform = TRANSFORM)
# DataLoader
TEST_LOADER = DataLoader(TEST_DATASET, batch_size = BATCH_SIZE, shuffle=False)
time_list= []

for idx, (batch, batch_path) in enumerate(TEST_LOADER):
    start_time = time.time()
    batch = batch.to(device)
    #batch_name = batch_path[0].split('/')[-1][:-4]
    with torch.no_grad():
        output_batch = generator_model(batch)
        output_mask = torch.sigmoid(output_batch)
    save_image(output_mask, checkpoint_dir = OUTPUT_DIR, image_type = batch_path)
    end_time = time.time()
    print(f'Performed inference within:{end_time - start_time} seconds')
    time_list.append(end_time - start_time)

time_array = np.array(time_list)
print(f'Mean inference time for a single batch of {BATCH_SIZE} images: {time_array.mean()}')
print(f'Mean inference time for a single image: {time_array.mean() / BATCH_SIZE}')

'''
Below is the example of wrong inference which can lead to 
incorrect results:

image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_tensor = torch.from_numpy(image_rgb).permute(2,0,1).float() / 255
image_tensor = torch.unsqueeze(image_tensor, dim=0)

image_tensor = image_tensor.to(device)
start_time = time.time()

with torch.no_grad():
	output = generator_model(image_tensor)
	output_mask = torch.sigmoid(output)

end_time = time.time()
print(f'Performed inference within:{end_time - start_time} seconds')
save_generated_image(output_mask, epoch = 300, step = 200, checkpoint_dir = OUTPUT_DIR, image_type = 'generated_mask')
'''

