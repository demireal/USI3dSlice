import numpy as np
import math
import cv2
import imageio
import scipy.io as spi

IN_DIM = (1, 50, 50)
PHI_MAX = 30
THETA_MAX = 45
X_SCALE = 0.5
Y_SCALE = 0.5
MASKSIZE = 400
DOWNSIZE_FACTOR = 4     # must either be 2, 4 or 8

INFILE = 'data/downsized_mats2_zeroed.mat'
MASKFOLDER = 'data'
INTERPOLATION = cv2.INTER_NEAREST

data = spi.loadmat(INFILE)['spliced_' + str(DOWNSIZE_FACTOR) + 'x']
data = np.flip(data, axis=1)

mask_size = int(MASKSIZE / DOWNSIZE_FACTOR)
mask = imageio.imread(MASKFOLDER + '/mask_' + str(DOWNSIZE_FACTOR) + 'x.png')[:, :, 1]
mask = np.uint8(mask / 255)

dims = data.shape
x0 = dims[0]
y0 = dims[1]
z0 = dims[2]


def get_bounding_box(theta, phi, dx, dy):
    # print('theta:',theta,'\tphi:',phi,'\tdx:',dx,'\tdy:',dy)
    h1 = [x0 / 2 - mask_size / 2 * math.sin(theta) + dx,  # + dist,##*math.cos(theta),
          y0 / 2 + mask_size / 2 * math.cos(theta) + dy]  ## + dist*math.sin(theta)]

    h2 = [x0 / 2 + mask_size / 2 * math.sin(theta) + dx,  # dist,##*math.cos(theta),
          y0 / 2 - mask_size / 2 * math.cos(theta) + dy]  ## + dist*math.sin(theta)]

    z_min = 0  # self.z0 / 2 - self.z0 / 2 * math.cos(phi)
    z_max = z0 * math.cos(phi)  # self.z0 / 2 + self.z0 / 2 * math.cos(phi)
    # print('h1:',h1,'\th2:',h2,'\tz_min:', z_min,'\tz_max:', z_max)
    return h1, h2, z_min, z_max


def get_slice(theta_n, phi_n, dx_n, dy_n):
    theta = math.radians(theta_n * THETA_MAX)
    phi = math.radians(phi_n * PHI_MAX)
    dx = X_SCALE * dx_n * x0 / 2  # +/- 200 pixels
    dy = Y_SCALE * dy_n * y0 / 2  # +/- 350 pixels

    # --- 1: Get bounding box dims ---
    h1, h2, z_min, z_max = get_bounding_box(theta=theta, phi=phi, dx=dx, dy=dy)
    w = mask_size
    h = mask_size

    # --- 2: Extract slice from volume ---
    # Get x_i and y_i for current layer
    x_offsets = np.linspace(z_min, z_max, h) * math.sin(phi) * math.cos(
        theta)  # np.linspace(-h/2, h/2, h) * math.sin(phi) * math.cos(theta)
    y_offsets = np.linspace(z_min, z_max, h) * math.sin(phi) * math.sin(
        theta)  # np.linspace(-h/2, h/2, h) * math.sin(phi) * math.sin(theta)

    # Tile and transpose
    x_offsets = np.transpose(np.tile(x_offsets, (w, 1)))
    y_offsets = np.transpose(np.tile(y_offsets, (w, 1)))

    x_i = np.tile(np.linspace(h1[0], h2[0], w), (h, 1))
    y_i = np.tile(np.linspace(h1[1], h2[1], w), (h, 1))

    x_i = np.array(np.rint(x_i + x_offsets), dtype='int')
    y_i = np.array(np.rint(y_i + y_offsets), dtype='int')

    # Don't forget to include the index offset from z!
    z_i = np.transpose(np.tile(np.linspace(z_min, z_max, h), (w, 1)))
    z_i = np.array(np.rint(z_i), dtype='int')

    # Flatten
    flat_inds = np.ravel_multi_index((x_i, y_i, z_i), (x0, y0, z0), mode='clip')

    # Fill in entire slice at once
    the_slice = np.take(data, flat_inds)

    # --- 3: Mask slice ---
    the_slice = np.multiply(the_slice, mask)
    return the_slice


t_arr = np.linspace(-1, 1, 201)
for index, t in enumerate(t_arr):
    print(index - 100)
    img = cv2.resize(get_slice(0, t, 0, 0), dsize=(300, 300), interpolation=INTERPOLATION)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

