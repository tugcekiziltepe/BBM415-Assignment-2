import cv2
from filters import mean_filter, gaussian_filter, kuwahara_filter

if __name__ == '__main__':

    image_path = 'images/france.jpg' # image path
    image = cv2.imread(image_path) # read image 

    window_size  = 3
    sigma = 1

    mean_filtered_image = mean_filter(image, window_size) # call mean filter function to filter the image
    cv2.imwrite("mean_filtered_image.jpg", mean_filtered_image) # save mean filtered image 

    kuwahara_filtered_image = kuwahara_filter(image, window_size) # call kuwahara filter function to filter the image
    cv2.imwrite("kuwahara_filtered_image.jpg", kuwahara_filtered_image) # save kuwahara filtered image

    gaussian_filtered_image = gaussian_filter(image,window_size, sigma) # call gaussian filter function to filter the image
    cv2.imwrite("gaussian_filtered_image.jpg", gaussian_filtered_image) # save gaussian filtered image

