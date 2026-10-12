from skimage.io import imread               # reading image files into numerical arrays
from skimage.filters import threshold_otsu  # otsu thresholding algorithm
import matplotlib.pyplot as plt             # imports the matplotlib's plotting module

car_image = imread("car.jpg", as_gray=True) # loading the car.jpg as a grayscale image
# should be a 2 dimensional array
print(car_image.shape)

# this next line is not required but, a grey scale pixel
# in skimage ranges between 0 & 1. multiplying it with 255
# will make it range between 0 & 255 (makes it something that users can relate with better)

gray_car_image = car_image * 255                    # multiplying the grayscale array values by 255, to make it a standard 0-255 pixel range
fig, (ax1, ax2) = plt.subplots(1,2)                 
ax1.imshow(gray_car_image, "gray")
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value
ax2.imshow(binary_car_image, cmap = "gray")
#plt.show()