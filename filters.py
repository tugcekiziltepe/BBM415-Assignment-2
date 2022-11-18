import numpy as np

def mean_filter(image, window_size):
    """
    It applies mean filter to given image.
    Parameters
    ----------
    image : numpy_darray
        given image
    window_size: int
        given windows size j as int and it forms j x j square filter matrix.
    Return
    ----------
    image : numpy_darray
        image is downsized to (n - window_size + 1) x (m - window_size + 1)
    """
    # amount_of_dowsize is the amount of downsize from one edge.
    # filter starts from this index and end with edge index - downsize
    downsize = (window_size - 1) // 2
    image_copy = image.copy()
    for dim in range(image_copy.shape[2]): # for each dimension (r, g, b) in image
        for y in range(downsize, image_copy.shape[0]- downsize): # from top to bottom
            for x in range(downsize, image_copy.shape[1]-downsize): # from left to right
                image_copy[y, x, dim] = int(np.mean(image_copy[y -downsize : y + downsize + 1, x - downsize : x + downsize + 1 , dim])) # get the mean of the window
    return image_copy[downsize:image_copy.shape[0]- downsize, downsize:image_copy.shape[1]- downsize]


def gaussian_filter(image, window_size, sigma):
    """
    It applies gaussian filter to given image.
    Parameters
    ----------
    image : numpy_darray
        given image
    window_size: int
        given windows size j as int and it forms j x j square filter matrix.
    sigma: float
        sigma is used as a parameter for gaussian function.
    Return
    ----------
    image : numpy_darray
        image is downsized to (n - window_size + 1) x (m - window_size + 1)
    """

    # amount_of_dowsize is the amount of downsize from one edge.
    # Filter starts from this index and end with edge index - downsize
    image_copy = image.copy()
    downsize = (window_size - 1 )// 2

    kernel = gaussian_kernel(window_size, sigma, downsize) # get gaussian kernel given window_size, sigma and downsize
    
    for dim in range(image_copy.shape[2]): # for each dimension (r, g, b) in image
        for y in range(downsize, image_copy.shape[0]- downsize): # from top to bottom
            for x in range(downsize, image_copy.shape[1]-downsize): # from left to right
                # sum of elementwise multiplication of the filter and the window is new pixel value
                image_copy[y, x, dim] = np.sum((np.multiply(image_copy[y - downsize : y + downsize + 1, x - downsize:x + downsize + 1 , dim], kernel)))

    return image_copy[downsize:image_copy.shape[0]- downsize, downsize:image_copy.shape[1]- downsize]

def gaussian_kernel(window_size, sigma, downsize):
    kernel = np.zeros((window_size, window_size), np.float64)

    for i in range(window_size):
        for j in range(window_size):
            constant = 1 / (2 * 3.14 * (sigma ** 2)) # constant in gaussian function
            kernel[j, i] = constant * np.exp((-(abs(j - downsize) ** 2 + abs(i - downsize)** 2)) / (2 * (sigma **2)))
    
    # This step is needed since given window size gaussian function do not give sum of 1.
    # If the sum is not equal to one, the image becomes darker.
    kernel = (kernel - kernel.min()) / (kernel - kernel.min()).sum() 

    # A convolution kernel is a correlation kernel that has been rotated 180 degrees.
    kernel = convolution_kernel(kernel)
    return kernel


def kuwahara_filter(image, window_size):
    """
    It applies kuwahara filter to given image.
    Parameters
    ----------
    image : numpy_darray
        given image
    window_size: int
        given windows size j as int and it forms j x j square filter matrix.
    Return
    ----------
    image : numpy_darray
        image is downsized to (n - window_size + 1) x (m - window_size + 1)

    """
    # amount_of_dowsize is the amount of downsize from one edge.
    # Filter starts from this index and end with edge index - downsize
    downsize = (window_size - 1 )// 2 
    image_copy = image.copy()
    # V(Value) from HSV color model
    v = rgb_to_v(image_copy)

    for y in range(downsize, image_copy.shape[0] - downsize):
        for x in range(downsize, image_copy.shape[1] - downsize):

            # Ranges of subregions
            Q1_r = y - downsize, y + 1, x - downsize, x + 1
            Q2_r = y - downsize, y + 1, x, x +  downsize + 1
            Q3_r = y, y + downsize + 1, x - downsize, x + 1
            Q4_r = y, y + downsize + 1, x, x + downsize + 1
            ranges = [Q1_r, Q2_r, Q3_r, Q4_r]

            # subregions are selected from V(Value)
            Q1 = v[ranges[0][0]: ranges[0][1], ranges[0][2]: ranges[0][3]]
            Q2 = v[ranges[1][0]: ranges[1][1], ranges[1][2]: ranges[1][3]]
            Q3 = v[ranges[2][0]: ranges[2][1], ranges[2][2]: ranges[2][3]]
            Q4 = v[ranges[3][0]: ranges[3][1], ranges[3][2]: ranges[3][3]]

            # calculate standard deviation of each subregion
            local_standard_deviations= np.array([np.std(Q1), np.std(Q2), np.std(Q3), np.std(Q4)])
            index = np.argmin(local_standard_deviations) # get index of minimum standard deviation

            # select subregion range which has minimum standard deviation
            image_copy[y, x, 0] = np.mean(image_copy[ranges[index][0]: ranges[index][1], ranges[index][2]:ranges[index][3], 0]) # update R dimension of the pixel
            image_copy[y, x, 1] = np.mean(image_copy[ranges[index][0]: ranges[index][1], ranges[index][2]:ranges[index][3], 1]) # update G dimension of the pixel
            image_copy[y, x, 2] = np.mean(image_copy[ranges[index][0]: ranges[index][1], ranges[index][2]:ranges[index][3], 2]) # update B dimension of the pixel

    return image_copy[downsize:image_copy.shape[0]- downsize, downsize:image_copy.shape[1]- downsize]


def rgb_to_v(image):
    """
    It applies gaussian filter to given image_copy.
    Parameters
    ----------
    image : numpy_darray
        given image
    Return
    ----------
    v : numpy_darray
        V dimension of HSV color model
    """
    image_copy = image.copy()
    v = np.zeros((image_copy.shape[0], image_copy.shape[1])) # initialize V array 

    for y in range(image_copy.shape[0]): # from top to bottom
        for x in range(image_copy.shape[1]): # from left to right

            # get rgb values for a pixel
            r, g, b = image_copy[y, x, 0] / 255.0, image_copy[y, x, 1]  / 255.0, image_copy[y, x, 1]  / 255.0
            
            # h, s, v = hue, saturation, value
            cmax = max(r, g, b)    # maximum of r, g, b

            # compute v
            v[y, x] = cmax * 100

    return  v

def convolution_kernel(filter):
    """
    A convolution kernel is a correlation kernel that has been rotated 180 degrees.
    Parameters
    ----------
    filter : numpy_darray
        given filter
    Return
    ----------
    filter : numpy_darray
        rotated filter
    """
    filter = np.flip(filter, axis=0)
    filter = np.flip(filter, 1)

    return filter