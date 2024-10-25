import gdspy
import cv2
import numpy as np

import warnings
warnings.filterwarnings('ignore')

IMAGE_PATH = 'images/epoch001_input_label.jpg'

image = cv2.imread(IMAGE_PATH)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)               #previously used 'numpy_img' instead of 'image_array'
ret, thresh = cv2.threshold(gray_image, 125, 255, cv2.THRESH_BINARY)
#thresh = cv2.resize(thresh, (1024, 1024), interpolation = cv2.INTER_AREA)


#previously used cv2.THRESH_BINARY_INV instead of cv2.THRESH_BINARY
# visualize the binary image
cv2.imshow('Thresholded image', thresh)
cv2.waitKey(0)


print(f'\n Image shape:{thresh.shape}')

def convert_to_bottom_left(coordinates, image_height):

  converted_coordinates = coordinates.copy()
  converted_coordinates[:, 1] = image_height - coordinates[:,1]

  return converted_coordinates

def intersection_point(seg1_start, seg1_end, seg2_start, seg2_end):
  """
  Finds the intersection point of two line segments
  :param seg1_start: A tuple (x,y) representing the starting point of segment 1
  :param seg1_end: A tuple (x,y) representing the ending point of segment 2
  :param seg2_start: A tuple (x,y) representing the starting point of segment 1
  :param seg2_end: A tuple (x,y) representing the ending point of segment 1
  :return: a tuple (x,y) representing the intersection point or None if no intersection exists
  """
  # calculate the direction vectors
  seg1_dir = (seg1_end[0] - seg1_start[0], seg1_end[1] - seg1_start[1])
  seg2_dir = (seg2_end[0] - seg2_start[0], seg2_end[1] - seg2_start[1])

  # check the parallel lines

  if seg1_dir[0] * seg2_dir[1] == seg1_dir[1] * seg2_dir[0]:
    return None
  # calculate line equation parameters for segment 1
  s1_x, s1_y = seg1_start
  a1, b1 = seg1_dir
  # calculate line equation parametes for segment 2
  s2_x, s2_y = seg2_start
  a2, b2 = seg2_dir

  # solve for intersection point coordinates (x,y)

  denominator = a1 * b2 - a2 * b1
  if denominator == 0:
    return None  # lines are coincident(might have and overlap)
  t = ((s2_x - s1_x) * b2 - (s2_y - s1_y) * a2) / denominator
  u = ((s1_x - s2_x) * b1 - (s1_y - s2_y) * a1) / denominator

  print(int(s1_x + t * a1), int(s1_y + t * b1))

  return (int(s1_x + t * a1), int(s1_y + t * b1))


width, height = thresh.shape[1], thresh.shape[0]

lib = gdspy.GdsLibrary()
gdspy.current_library=gdspy.GdsLibrary()

layerNum = 3


gridcell = lib.new_cell('GRID')

#thresh = cv2.resize(thresh, (2048, 2048), interpolation = cv2.INTER_AREA)

contours1, _ = cv2.findContours(image = thresh, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)

if len(contours1) == 1:
  correct_format = np.array(contours1).squeeze()
  correct_format_copy = correct_format.copy()
  counter = 0

  #code for inserting the missed concave corner point

  for j in range(correct_format.shape[0] - 1):
    if np.linalg.norm(correct_format[j] - correct_format[j + 1]) > 1:

        seg1_start = tuple(correct_format[j-1])
        seg1_end = tuple(correct_format[j])

        seg2_start = tuple(correct_format[j+2])
        seg2_end = tuple(correct_format[j+1])


        intersection_x, intersection_y = intersection_point(seg1_start, seg1_end, seg2_start, seg2_end)
        correct_format_copy = np.insert(correct_format_copy, j + 1 + counter,list(intersection_x, intersection_y), axis = 0 )
        counter += 1


  #correct_format = contours1.reshape((contours1.shape[0],2))
  bottom_left_coord = convert_to_bottom_left(correct_format_copy, image_height = 1024)  #2048
  final_coord = [(i[0] * 1 / 1000,i[1] * 1 / 1000) for i in bottom_left_coord]

  figure = gdspy.Polygon(final_coord)             #layer=int(layerNum)
  expanded_figure = gdspy.offset(figure, 0.001, layer = int(layerNum) )
  gridcell.add(expanded_figure)                   #figure
else:
  for sub_contour in contours1:
    #correct_format = sub_contour.reshape((sub_contour.shape[0],2))
    correct_format = sub_contour.squeeze()
    correct_format_copy = correct_format.copy()

    counter = 0

    # #code for inserting the missed concave corner point
    for j in range(correct_format.shape[0] - 1):
      if np.linalg.norm(correct_format[j] - correct_format[j + 1]) > 1:

          seg1_start = tuple(correct_format[j-1])
          seg1_end = tuple(correct_format[j])

          seg2_start = tuple(correct_format[j+2])
          seg2_end = tuple(correct_format[j+1])

          if intersection_point(seg1_start, seg1_end, seg2_start, seg2_end) == None:

            continue

          intersection_x, intersection_y = intersection_point(seg1_start, seg1_end, seg2_start, seg2_end)


          correct_format_copy = np.insert(correct_format_copy, j + 1 + counter,[intersection_x, intersection_y], axis = 0 )
          counter += 1

    bottom_left_coord = convert_to_bottom_left(correct_format_copy, image_height = 1024)  #2048
    final_coord = [(i[0] * 1 / 1000,(i[1] + 1) * 1 / 1000) for i in bottom_left_coord]

    correct_format = [(a/1000,b/1000) for i in sub_contour for a,b in i ]
    figure = gdspy.Polygon(final_coord)             #layer=int(layerNum)
    expanded_figure = gdspy.offset(figure, 0, layer = int(layerNum) )
    gridcell.add(figure)


lib.write_gds("topologies/input_label.gds")
