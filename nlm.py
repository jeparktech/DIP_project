import numpy as np
import cv2
from pathlib import Path

# Define the paths to the directories for noisy and denoised images.
noised_images_directory = Path('./Datasets/Set12/noisedImages')
denoised_images_directory = Path('./Datasets/Set12/denoisedImages')

# Create the 'denoisedImages' directory if it doesn't exist.
denoised_images_directory.mkdir(exist_ok=True)

# Dictionary to keep track of the denoised image filenames.
denoised_images_info = {}

# Get the list of noisy image filenames from the 'noisedImages' directory.
noisy_image_filenames = [f.name for f in noised_images_directory.glob('*.png')]

# Apply Non-Local Means Denoising to each noisy image and save the result.
for noisy_filename in noisy_image_filenames:
    # Read the noisy image from the 'noisedImages' folder
    noisy_image_path = noised_images_directory / noisy_filename
    noisy_image = cv2.imread(str(noisy_image_path), cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully.
    if noisy_image is not None:
        # Apply Non-Local Means Denoising.
        denoised_image = cv2.fastNlMeansDenoising(noisy_image, None, 3.0, 7, 21)

        # Generate the denoised image filename and path.
        denoised_filename = noisy_filename.replace('noisy', 'denoised')
        denoised_image_path = denoised_images_directory / denoised_filename

        # Save the denoised image in the 'denoisedImages' folder.
        cv2.imwrite(str(denoised_image_path), denoised_image)

        # Update the dictionary with the path for the denoised image.
        denoised_images_info[noisy_filename] = str(denoised_image_path.relative_to(denoised_images_directory.parent))

# Return the dictionary with noisy and denoised filenames, including their new paths.
denoised_images_info
