import os
import cv2
import csv
import pywt
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import peak_signal_noise_ratio

# PSNR calculation
def calculate_psnr(original_image, denoised_image):
    return compare_psnr(original_image, denoised_image)

# SSIM calculation
def calculate_ssim(original_image, denoised_image):
    if (len(original_image.shape) == 3):
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
    return compare_ssim(original_image, denoised_image)
# CW-SSIM calculation
def calculate_cw_ssim(original_image, denoised_image):
    if (len(original_image.shape) == 3):
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

    # Define the wavelet function
    wavelet = 'db1'
    # Compute the wavelet coefficients
    coeffs2_original = pywt.dwt2(original_image, wavelet, 'periodization')
    coeffs2_denoised = pywt.dwt2(denoised_image, wavelet, 'periodization')
    # Extract the LL subband
    LL1 = coeffs2_original[0]
    LL2 = coeffs2_denoised[0]
    # The data range for CW-SSIM might be different, depending on the range of your wavelet coefficients.
    # You may need to adjust the data_range value accordingly.
    data_range = LL1.max() - LL1.min()
    # Compute the CW-SSIM score
    cw_ssim_score = compare_ssim(LL1, LL2, data_range=data_range)
    return cw_ssim_score

OUTPUT_PATH = './Outputs'
result_dict = {}
def process_folder(dataset, method):
    if method == "bilateral":
        csv_filename = os.path.join(OUTPUT_PATH, "Bilateral_results_python.csv")
        denoised_img = "Bilateral.png"
    elif method == "NLM10":
        csv_filename = os.path.join(OUTPUT_PATH, "NLM10_results_python.csv")
        denoised_img = "NLM10.png"
    elif method == "NLM20":
        csv_filename = os.path.join(OUTPUT_PATH, "NLM20_results_python.csv")
        denoised_img = "NLM20.png"
    elif method == "NLM30":
        csv_filename = os.path.join(OUTPUT_PATH, "NLM30_results_python.csv")
        denoised_img = "NLM30.png"
    else:
        csv_filename = ""
        denoised_img = ""


    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Dataset', 'Challenge', 'Noise Type', 'PSNR', 'SSIM', 'CW_SSIM']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Process each subfolder
        subfolders = [f.path for f in os.scandir(OUTPUT_PATH) if f.is_dir()]
        for subfolder in subfolders:
            if dataset in subfolder:
                if dataset == 'CURE-OR':
                    image_folders = [f.path for f in os.scandir(subfolder) if f.is_dir()]
                    for image_folder in image_folders:
                        folder_name = os.path.basename(image_folder)
                        noise_type = result_dict[str(int(folder_name))]['challenge_type']
                        challenge = result_dict[str(int(folder_name))]['challenge_level']

                        denoised_image_path = os.path.join(image_folder, denoised_img)
                        gt_image_path = os.path.join(image_folder, "GT.png")

                        denoised_image = cv2.imread(denoised_image_path)
                        gt_image = cv2.imread(gt_image_path)
                        psnr = calculate_psnr(gt_image, denoised_image)
                        ssim = calculate_ssim(gt_image, denoised_image)
                        cw_ssim = calculate_cw_ssim(gt_image, denoised_image)
                        writer.writerow(
                            {'Dataset': dataset, 'Challenge': challenge, 'Noise Type': noise_type, 'PSNR': psnr,
                             'SSIM': ssim, 'CW_SSIM': cw_ssim})

                elif dataset == 'CURE-TSR':
                    image_folders = [f.path for f in os.scandir(subfolder) if f.is_dir()]
                    for image_folder in image_folders:
                        folder_name = os.path.basename(image_folder)
                        comp = folder_name.split('_')
                        noise_type = comp[1]
                        challenge = comp[2]

                        denoised_image_path = os.path.join(image_folder, denoised_img)
                        gt_image_path = os.path.join(image_folder, "GT.png")

                        denoised_image = cv2.imread(denoised_image_path)
                        gt_image = cv2.imread(gt_image_path)
                        psnr = calculate_psnr(gt_image, denoised_image)
                        ssim = calculate_ssim(gt_image, denoised_image)
                        cw_ssim = calculate_cw_ssim(gt_image, denoised_image)
                        writer.writerow({'Dataset': dataset, 'Challenge': challenge, 'Noise Type': noise_type, 'PSNR': psnr,
                                         'SSIM': ssim, 'CW_SSIM': cw_ssim})

                elif dataset == 'SIDD':
                    image_folders = [f.path for f in os.scandir(subfolder) if f.is_dir()]
                    for image_folder in image_folders:
                        denoised_image_path = os.path.join(image_folder, denoised_img)
                        gt_image_path = os.path.join(image_folder, "GT.png")

                        denoised_image = cv2.imread(denoised_image_path)
                        gt_image = cv2.imread(gt_image_path)
                        psnr = calculate_psnr(gt_image, denoised_image)
                        ssim = calculate_ssim(gt_image, denoised_image)
                        cw_ssim = calculate_cw_ssim(gt_image, denoised_image)
                        writer.writerow({'Dataset': dataset, 'Challenge': -1, 'Noise Type': -1, 'PSNR': psnr,
                                         'SSIM': ssim, 'CW_SSIM': cw_ssim})




            # images = os.listdir(subfolder)
            # gt_path = os.path.join(subfolder, 'GT.png')
            #
            # # Process each denoised image
            # for image_name in ['Bilateral.png', 'NLM10.png', 'NLM20.png', 'NLM30.png']:
            #     image_path = os.path.join(subfolder, image_name)
            #     psnr_value = calculate_psnr(gt_path, image_path)
            #
            #     # Write results to CSV
            #     writer.writerow({'Image': image_name, 'PSNR': psnr_value})

process_folder("SIDD", "bilateral")
process_folder("SIDD", "NLM10")
process_folder("SIDD", "NLM20")
process_folder("SIDD", "NLM30")


process_folder("CURE-TSR", "bilateral")
process_folder("CURE-TSR", "NLM10")
process_folder("CURE-TSR", "NLM20")
process_folder("CURE-TSR", "NLM30")

CURE_OR_CSV_PATH = './Datasets/mini-CURE-OR/train.csv'

def read_csv_and_create_dict(csv_file_path):
    data_dict = {}

    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)

        for row in csv_reader:
            image_id = row['imageID']
            challenge_type = row['challengeType']
            challenge_level = row['challengeLevel']

            # Create a dictionary entry with image ID as the key and other columns as the value
            data_dict[image_id] = {
                'challenge_type': challenge_type,
                'challenge_level': challenge_level
            }

    return data_dict

result_dict = read_csv_and_create_dict(CURE_OR_CSV_PATH)

process_folder("CURE-OR", "bilateral")
process_folder("CURE-OR", "NLM10")
process_folder("CURE-OR", "NLM20")
process_folder("CURE-OR", "NLM30")