import os
import cv2
import numpy as np
from skimage.restoration import wiener, denoise_bilateral
from scipy.signal import convolve2d
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


CURE_OR_path_train = './Datasets/mini-CURE-OR/train'

image_files = sorted(os.listdir(CURE_OR_path_train))

CURE_OR_image_dict = {}

for i in range(len(image_files)):
    image_path = os.path.join(CURE_OR_path_train, image_files[i])
    is_ground_truth = i < 150
    base_name = os.path.splitext(image_files[i])[0]
    preprocessed_image = preprocess_image(image_path)


    # Add the preprocessed image to the dictionary
    if is_ground_truth:
        CURE_OR_image_dict[base_name] = {'gt': preprocessed_image, 'noised': []}
    else:
        gt_name = f'{int(base_name) %150:05d}'
        CURE_OR_image_dict[gt_name]['noised'].append([base_name, preprocessed_image])

# Set up subplots for original, noisy, and denoised images
# fig, axs = plt.subplots(len(CURE_OR_image_dict), 3, figsize=(15, 5 * len(CURE_OR_image_dict)))

# Iterate through the examples in CURE_OR_image_dict
# for i, (example_name, images_dict) in enumerate(CURE_OR_image_dict.items()):
#
#     # Access the ground truth and noised images for the example
#     gt_image = images_dict['gt']
#     noised_images = images_dict['noised']
#
#     for noised_image in noised_images:
#         denoised_image = nlm_denoising(noised_image)
#
#         plt.subplot(131)
#         plt.imshow(gt_image)
#
#         plt.subplot(132)
#         plt.imshow(noised_image)
#
#         plt.subplot(133)
#         plt.imshow(denoised_image)


    #
    # # Plot the images
    # axs[i, 0].imshow(gt_image, cmap='gray')
    # axs[i, 0].set_title('Original Image')
    # axs[i, 0].axis('off')
    #
    # # Randomly choose one noised image for display
    # noised_image = np.random.choice(noised_images)
    # axs[i, 1].imshow(noised_image, cmap='gray')
    # axs[i, 1].set_title('Noised Image')
    # axs[i, 1].axis('off')
    #
    # # Apply NLM denoising to the chosen noised image
    # denoised_image = nlm_denoising(noised_image)
    # axs[i, 2].imshow(denoised_image, cmap='gray')
    # axs[i, 2].set_title('Denoised Image')
    # axs[i, 2].axis('off')



# Example usage:
# Access the ground truth image and corresponding noised images
# example_ground_truth = CURE_OR_image_dict['00002']['gt']
# example_noised_image = CURE_OR_image_dict['00002']['noised'][15][1] #total of 65 noised image per ground truth.
# example_denoised_image = nlm_denoising(example_noised_image, 10)
# #
# plt.subplot(131)
# plt.imshow(example_ground_truth)
# plt.axis('off')
#
# plt.subplot(132)
# plt.imshow(example_noised_image)
# plt.axis('off')
#
# plt.subplot(133)
# plt.imshow(example_denoised_image)
# plt.axis('off')
# plt.show()

OUTPUT_PATH = './Outputs/CURE-OR'


for gt_file_name, values in CURE_OR_image_dict.items():
    gt_image = values['gt']
    gt_image = Image.fromarray(np.uint8(gt_image * 255.0))
    noised_images = values['noised']
    for noised_file_name, noised_image in noised_images:
        denoised_bilateral = bilateral(noised_image)
        denoised_NLM10 = nlm_denoising(noised_image, 10)
        denoised_NLM20 = nlm_denoising(noised_image, 20)
        denoised_NLM30 = nlm_denoising(noised_image, 30)

        subfolder_path = os.path.join(OUTPUT_PATH, noised_file_name)
        os.makedirs(subfolder_path, exist_ok=True)

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















