import numpy as np

img_width = 320
img_height = 240
fx = 241.42
fy = 241.42

def world2pixel(outputs, img_width, img_height, fx, fy):
        x, y, z = outputs
        p_x = x * fx / z + img_width / 2
        p_y = img_height / 2 - y * fy / z
        return p_x, p_y

outputs = np.loadtxt('test_res.txt')
outputs = outputs.reshape(outputs.shape[0], 21, 3)
pixel_output = np.zeros(shape = (8496, 21, 2))
for i in range(8496):
        for j in range(21):
                pixel_output[i][j] = world2pixel(outputs[i][j], img_width, img_height, fx, fy)

np.savetxt('test_res_pixel.txt', pixel_output.reshape(pixel_output.shape[0], -1))
