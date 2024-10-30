import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image, ImageOps
import glob as glob
from gdsast import *
import argparse
import os
import tqdm

def delete_empty_files(directory):
    count = 0
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and os.stat(file_path).st_size == 0:
                os.remove(file_path)
                count += 1
                print('Deleted empty file:',filename)
    except (FileNotFoundError, PermissionError) as e:
        print("Error:", e)
    return count

class Topology:
  def __init__(self, raw_dict):
    self.raw_dict = raw_dict

  def create_layer_ref(self):
    layer_dict = {}

    for element in self.raw_dict:
      curr_layer = element['layer']

      if curr_layer not in layer_dict.keys():
        batch_list = [element['xy']]
        layer_dict[curr_layer] = [batch_list]
      elif curr_layer in layer_dict.keys():
        batch_list = [element['xy']]
        layer_dict[curr_layer].append(batch_list)

    return layer_dict

  def max_coords(self, raw_layer_number):
    '''
    here we want to extract the maximum x and y coordiantes from
    the layers we have in our Topology.

    for example we have only 2 layers:
    number '1' refers to ground truth layer
    number '2' refers to corrected layer
    to extract a concrete layer from 'layer dict'
    we simply use layer numbers as keys:

    layer_dict[1] -> extract first layer
    layer_dict[2] -> extract second layer

    then converting the list of lists to numpy array for each layer and find
    maximum elements
    '''
    layer_dict = self.create_layer_ref()
    # in case we want to output the max/min element for random layer
    # random_layer_number = random.choice([ element for element in layer_dict.keys()] )
    # second_layer = np.hstack(layer_dict[random_layer_number]).reshape((-1, 2))

    second_layer = np.hstack(layer_dict[raw_layer_number]).reshape((-1, 2))

    max_x = max(second_layer[:, 0])
    max_y = max(second_layer[:, 1])
    min_x = min(second_layer[:, 0])
    min_y = min(second_layer[:, 1])
    return (min_x, min_y), (max_x, max_y)

  def single_layer_list(self, layer_number):
    layer_dict = self.create_layer_ref()
    layer_coord = np.hstack(layer_dict[layer_number]).reshape((-1, 2))

    return layer_coord


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Enter path to folder with gds files and layer number to create an image')
  parser.add_argument('input_folder_path', type = str, help = 'Path for input folder with gdsii files')
  parser.add_argument('output_folder_path', type = str, help = 'Path for output folder with images')
  parser.add_argument('layer_number', type = int, help = 'Number of layer in topology file; 3 - refers to original, 100 - refers to correction  ' )

  args = parser.parse_args()
  layers = args.layer_number

  input_folder_path = args.input_folder_path
  input_folder_files = input_folder_path + '/*.gds'
  layer_number = args.layer_number
  output_folder_path = args.output_folder_path

  if not os.path.exists(output_folder_path):
    os.mkdir(output_folder_path)

  if layer_number == 3:
      subfolder_path = os.path.join(output_folder_path, 'origin')
      if not os.path.exists(subfolder_path):
          os.mkdir(subfolder_path)
  elif layer_number == 100:
      subfolder_path = os.path.join(output_folder_path, 'correction')
      if not os.path.exists(subfolder_path):
          os.mkdir(subfolder_path)

  deleted_files_count = delete_empty_files(input_folder_path)
  print("Total empty files deleted:",deleted_files_count)

  folder_files = glob.glob(input_folder_files)
  folder_files.sort()
  print('Number of topology files in a folder:',len(folder_files))
  print(folder_files)

  for file in tqdm.tqdm(folder_files):
      with open(file, "rb") as f:
          gds = gds_read(f)

      file_name = file.split('/')[-1].split('.')[0]
      topol_elements = gds['structures'][0]['elements']

      test_top = Topology(topol_elements)
      k = test_top.create_layer_ref()
      # uncomment the string below to see the existing layers in topology
      # print(f'Here is the list of exiting layers in topology:{k.keys()}')

      '''
      below go two critically important rows that must retain 
      '''
      fig, ax = plt.subplots(layout='constrained')
      fig.set_constrained_layout_pads(w_pad=0, h_pad=0)

      plt.style.use('dark_background')
      fig.set_size_inches(10, 10)
      fig.set_dpi(102.4)
      # print(fig.dpi)           #20.48, 9.70
      for element in k[layer_number]:
          # Create the polygon and add it to a plot
          poly = Polygon(*element, facecolor='white', antialiased=False)  # *element
          ax.add_patch(poly)

      # Set the plot limits and show the plot
      ax.set_xlim([488, 1512])  # 515 1485
      ax.set_ylim([488, 1512])
      ax.axis('off')

      centered_name = subfolder_path + '/' + file_name + '_centered.jpg'
      print(subfolder_path, file_name)
      plt.savefig(centered_name, bbox_inches='tight', pad_inches=0, dpi=102.4)
      plt.close()



