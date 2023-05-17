import pywt
import numpy as np
import cv2
import math
from skimage.metrics import peak_signal_noise_ratio
titles = ['noise', ' denoise ']

def add_gaussian_noise(image, mean, std_dev):
    # Generate Gaussian noise
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
    # Add noise to the image
    noisy_image = cv2.add(image, noise)
    return noisy_image

def denoise_image(image,lev):
    # Convert the image to grayscale

    # Apply the DWT
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs

    if (1 == lev):
        coeffs = pywt.dwt2(image, 'haar')
        cA, (cH, cV, cD) = coeffs
    else:
        for i in range(1,  lev+1 ):
            cA, (cH, cV, cD) = coeffs
            coeffs = pywt.dwt2(cA, 'haar')



   # Set a threshold value for coefficients
    threshold =50

    #Threshold the detail coefficients (cH, cV, cD)
    cH_thresh = pywt.threshold(cH, threshold)
    cV_thresh = pywt.threshold(cV, threshold)
    cD_thresh = pywt.threshold(cD, threshold)

    # Reconstruct the denoised image

    coeffs=(cA,(cH_thresh,cV_thresh,cD_thresh))
    denoised_image = pywt.idwt2(coeffs ,'haar')
    # Convert the image back to the original color space

    return denoised_image

def calculate_snr(original, noise ):
    # Calculate the mean square error (MSE) between the original and denoised images
    mse = np.mean((original - noise) ** 2)

    # Calculate the peak signal-to-noise ratio (PSNR) using the MSE
    psnr = 20 * math.log10(np.max(original) / math.sqrt(mse))

    # Calculate the signal-to-noise ratio (SNR) using the mean square error (MSE) between the original and noisy images
    snr = 20 * math.log10(np.mean(original) / math.sqrt(mse))

    return psnr, snr



#######################################################################
# main function                                                       #
#######################################################################

# Load the image
image = cv2.imread('E:\learning\DSP_project_code\picture\cat.jpg')

# Define the mean and standard deviation of the Gaussian noise
mean = 10000
std_dev = 10
level = 1

# Add Gaussian noise to the image
noisy_image = add_gaussian_noise(image, mean, std_dev)
b, g, r = cv2.split(noisy_image)

# Perform noise reduction
denoised_b = denoise_image(b,level)
denoised_g = denoise_image(g,level)
denoised_r = denoise_image(r,level)
denoised_image = cv2.merge((denoised_b, denoised_g, denoised_r))


#Calculate SNR between the original, noisy, and denoised images
if 1 == level:
    print('PSNR value: {}'.format(peak_signal_noise_ratio(image, noisy_image)))
    print('Denoise PSNR value: {}'.format(peak_signal_noise_ratio(image, denoised_image.astype(np.uint8))))


# Display the original and denoised images
cv2.imshow('Original Image', image)
cv2.imshow('noised Image', noisy_image)
cv2.imshow('Denoised Image', denoised_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
