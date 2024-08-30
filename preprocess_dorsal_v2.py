import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def normalize_data(x, low=0, high=1, data_type=None):
	x = np.asarray(x, dtype='float32')
	min_x, max_x = np.min(x), np.max(x)
	x = x - float(min_x)
	x = x / float((max_x - min_x))
	x = x * (high - low) + low
	if data_type is None:
		return np.asarray(x, dtype='float32')
	return np.asarray(x, dtype=data_type)


def remove_hair(image, mexican_kernel_size, low=1, high=4):

	try:
		read_kernel = cv2.imread(f'./MexicanHatKernelData/Kernel_{mexican_kernel_size}.jpg', 0)
	#print(read_kernel)
	except FileNotFoundError:
		print('please choose correct size of kernel')

	normalized_kernel = normalize_data(read_kernel, low, high)
	hair_remove = convolve2d(image, normalized_kernel, mode='same', fillvalue=0)
	return hair_remove


def preprocessing(img):
    removed = remove_hair(img, 11)
    img = cv2.convertScaleAbs(removed, alpha=255/removed.max())
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lengths = [len(_) for _ in contours]
    contours = [_ for _ in contours if len(_) == max(lengths)]
    contours = contours[0]

    img = cv2.fastNlMeansDenoising(img)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (5, 5), 10)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 6)
    img = cv2.drawContours(
        img, [contours], 0, (255, 0, 0), 2)
    return np.bitwise_not(img)


if __name__ == '__main__':
    img = cv2.imread('./dorsal_git/person_111_db2_L3.png', 0)
    processed_img = preprocessing(img)
    plt.imshow(img, cmap='gray'), plt.show()
    plt.imshow(processed_img, cmap='gray'), plt.show()