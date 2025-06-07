import time

import cv2
import gdspy
import numpy as np

start_time = time.time()
INPUT_PATH = "inference/input_img/cell19_padded_synthesized_image.jpg"
OUTPUT_PATH = "data/external/synthesized_mask_19.gds"

dx = 515 - 27
dy = 515 - 28


image = cv2.imread(INPUT_PATH)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_image, 125, 255, cv2.THRESH_BINARY)

"""
Также можно сделать resize изображения и отобразить картинку:
thresh = cv2.resize(thresh, (1024, 1024), interpolation = cv2.INTER_AREA)

Отобразить картинку после применния пороговой фильтрации
visualize the binary image
cv2.imshow('Thresholded image', thresh)
cv2.waitKey(0)
"""

# Ищем индексы белых пикселей на изображении, по сути их координаты
white_pix_indices = list((np.where(thresh == 255)))

lib = gdspy.GdsLibrary(unit=1e-9)
cell = lib.new_cell("REF_CELL")  # добавили примитивную/референсную ячейку в библиотеку
top_cell = lib.new_cell("TOP_CELL")  # добавили топовую ячейку в библиотеку
cell.add(
    gdspy.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)], layer=5)
)  # добавили референсный полигон

# добавяляем в топовую ячейку примитивные полигоны размером 1x1 нм^2
# переводим координаты белых пикселей из домена top_left в bottom_right
for _, poly_position in enumerate(
    list(zip(white_pix_indices[1], 1024 - white_pix_indices[0]))
):
    top_cell.add(gdspy.CellReference(cell, origin=poly_position))

top_cell_polygons = top_cell.get_polygonsets()
# Создаём ячейку union, в которой будут объединены все полигоны из ячейки top_cell
union_cell = lib.new_cell("UNION_CELL")
for i in range(len(top_cell_polygons)):
    if i == 0:
        union_figure = gdspy.boolean(
            top_cell_polygons[i], top_cell_polygons[i + 1], operation="or", layer=3
        )
    else:
        union_figure = gdspy.boolean(
            union_figure, top_cell_polygons[i], operation="or", layer=3
        )

# union_figure = gdspy.boolean(top_cell_polygons, operation='or', layer=3)
union_cell.add(union_figure.translate(dx, dy))

# Удаляем ненужные ячейки
lib.remove(top_cell)
lib.remove(cell)


lib.write_gds(OUTPUT_PATH)
end_time = time.time()

print(f"Image converted to topology within {round(end_time - start_time)} seconds ")
print(f"Saved result to {OUTPUT_PATH}")
