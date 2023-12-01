import os
import cv2
import numpy as np
from skimage.restoration import wiener, denoise_bilateral
from PIL import Image
from scipy.signal import convolve2d
from skimage import color
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SIDD_PATH = './Datasets/SIDD_Small_sRGB_Only/Data'

def preprocess_image(image_path, target_size=(256, 256), normalize=True):
    # Read the image
    image = cv2.imread(image_path)

    # Resize the image to the target size
    image = cv2.resize(image, target_size)

    # Convert the image to float32
    image = image.astype(np.float32)

    # Normalize the pixel values to the range [0, 1] if requested
    if normalize:
        image /= 255.0

    return image

def nlm_denoising(image, height):
    image_8u = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.fastNlMeansDenoising(image_8u, None, height, 7, 21)

def bilateral(image):
    if len(image.shape) == 3:
        channel_axis = -1
    else:
        channel_axis = None

    denoised_image = denoise_bilateral(image, sigma_color=0.05, sigma_spatial=15, channel_axis=channel_axis)

    return denoised_image
# Initialize an empty dictionary to store ground truth and noised images
image_dict = {}

# Iterate through each folder in the dataset folder
for folder in os.listdir(SIDD_PATH):
    folder_path = os.path.join(SIDD_PATH, folder)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Find all files in the folder
        files = os.listdir(folder_path)

        # Filter out images based on the naming convention
        ground_truth = [f for f in files if f.startswith("GT")]
        noised_image = [f for f in files if f.startswith("NOISY")]

        # Ensure there is one ground truth and one noised image in the folder
        if len(ground_truth) == 1 and len(noised_image) == 1:
            ground_truth_path = os.path.join(folder_path, ground_truth[0])
            noised_image_path = os.path.join(folder_path, noised_image[0])

            # Add to the dictionary
            image_dict[folder] = {
                'ground_truth': ground_truth_path,
                'noised_image': noised_image_path
            }

OUTPUT_PATH = './Outputs/SIDD'
l = len(image_dict)
k = 0
for folder, images in image_dict.items():
    print(f"{k}/{l}")
    k += 1
    ground_truth_image = preprocess_image(images['ground_truth'])
    noised_image = preprocess_image(images['noised_image'])

    denoised_bilateral = bilateral(noised_image)
    denoised_NLM10 = nlm_denoising(noised_image, 10)
    denoised_NLM20 = nlm_denoising(noised_image, 20)
    denoised_NLM30 = nlm_denoising(noised_image, 30)

    subfolder_path = os.path.join(OUTPUT_PATH, folder)
    os.makedirs(subfolder_path, exist_ok=True)

    gt_image = Image.fromarray(np.uint8(ground_truth_image * 255.0))
    gt_image_path = os.path.join(subfolder_path, f"GT.png")

    noise_image_path = os.path.join(subfolder_path, f"Noised.png")
    noise_image = Image.fromarray(np.uint8(noised_image * 255.0))

    denoised_bilateral_path = os.path.join(subfolder_path, f"Bilateral.png")
    denoised_bilateral_image = Image.fromarray(np.uint8(denoised_bilateral * 255.0))

    denoised_NLM10_path = os.path.join(subfolder_path, f"NLM10.png")
    denoised_NLM10_image = Image.fromarray(np.uint8(denoised_NLM10))

    denoised_NLM20_path = os.path.join(subfolder_path, f"NLM20.png")
    denoised_NLM20_image = Image.fromarray(np.uint8(denoised_NLM20))

    denoised_NLM30_path = os.path.join(subfolder_path, f"NLM30.png")
    denoised_NLM30_image = Image.fromarray(np.uint8(denoised_NLM30))

    # save images
    gt_image.save(gt_image_path)
    noise_image.save(noise_image_path)
    denoised_bilateral_image.save(denoised_bilateral_path)
    denoised_NLM10_image.save(denoised_NLM10_path)
    denoised_NLM20_image.save(denoised_NLM20_path)
    denoised_NLM30_image.save(denoised_NLM30_path)





# for folder, images in image_dict.items():
#     ground_truth_image = preprocess_image(images['ground_truth'])
#     noised_image_image = preprocess_image(images['noised_image'])
#
#     # Display images side by side
#     plt.figure(figsize=(8, 4))
#
#     plt.subplot(1, 2, 1)
#     plt.imshow(ground_truth_image)
#     plt.title('Ground Truth')
#     plt.axis('off')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(noised_image_image)
#     plt.title('Noised Image')
#     plt.axis('off')
#
#     plt.show()
