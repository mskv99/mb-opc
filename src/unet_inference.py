import os
import cv2
import torch
import argparse
import time
from models.model import *
from data.dataset import *
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description = 'Performing inference on topology images')
parser.add_argument('inference_folder', type = str, help = 'Relative path to an inference image folder')
args = parser.parse_args()

DATA_PATH = args.inference_folder # 'input_img/cell15_padded_input_label.jpg'
MODEL_PATH = 'checkpoints/exp_2/last_checkpoint.pth'
OUTPUT_DIR = 'inference/output_img'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running inference on device: {device}')

def save_generated_image(output, epoch, step, checkpoint_dir="checkpoints", image_type='true_correction'):
    # # Convert from [-1, 1] to [0, 255] for saving
    # output = 0.5 * (output + 1)  # Brings values to [0, 1] range

    single_image = torch.mean(output[0], axis=0)
    single_image[single_image > 0.5] = 1.0
    single_image[single_image <= 0.5] = 0.0

    img_save_path = os.path.join(checkpoint_dir, f"{image_type}_epoch{epoch}_step{step}.png")
    cv2.imwrite(f"{img_save_path}", (single_image*255).detach().cpu().numpy())
    # saved_img = Image.open(img_save_path)
    # saved_img.show()
    print(f"Saved generated image at {img_save_path}")

generator_model = Generator(in_ch = 3, out_ch = 3)
# generator_model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
generator_model.load_state_dict(torch.load(MODEL_PATH, map_location = device)['model_state_dict'])
generator_model = generator_model.to(device)
generator_model.eval()
print('Model initialized:', generator_model)

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

# Dataset paths
# test_dataset = OPCDataset("data/processed/gds_dataset_new/origin/test_origin", "data/processed/gds_dataset_new/correction/test_correction", transform=transform)
test_dataset = TestDataset(DATA_PATH, transform=transform)

# DataLoader
TEST_LOADER = DataLoader(test_dataset, batch_size=1, shuffle=True)

for idx, image in enumerate(TEST_LOADER):
    start_time = time.time()
    image = image.to(device)
    with torch.no_grad():
        output = generator_model(image)
        output_mask = torch.sigmoid(output)
    save_generated_image(output_mask, epoch=1, step=idx, checkpoint_dir=OUTPUT_DIR, image_type='generated_mask')
    end_time = time.time()
    print(f'Performed inference within:{end_time - start_time} seconds')



# image = cv2.imread(IMAGE_PATH)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# image_tensor = torch.from_numpy(image_rgb).permute(2,0,1).float() / 255
# image_tensor = torch.unsqueeze(image_tensor, dim=0)
#
# image_tensor = image_tensor.to(device)
# start_time = time.time()
#
# with torch.no_grad():
# 	output = generator_model(image_tensor)
# 	output_mask = torch.sigmoid(output)
#
# end_time = time.time()
# print(f'Performed inference within:{end_time - start_time} seconds')
# save_generated_image(output_mask, epoch = 300, step = 200, checkpoint_dir = OUTPUT_DIR, image_type = 'generated_mask')


