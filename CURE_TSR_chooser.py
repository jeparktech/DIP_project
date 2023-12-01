import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import wiener, denoise_bilateral
import matplotlib.image as mpimg
from PIL import Image

CURE_TSR_GT_PATH = './Datasets/CURE-TSR/Real_Train/ChallengeFree'

CURE_TSR_DL1_PATH = './Datasets/CURE-TSR/Real_Train/DirtyLens-1'
CURE_TSR_DL2_PATH = './Datasets/CURE-TSR/Real_Train/DirtyLens-2'
CURE_TSR_DL3_PATH = './Datasets/CURE-TSR/Real_Train/DirtyLens-3'
CURE_TSR_DL4_PATH = './Datasets/CURE-TSR/Real_Train/DirtyLens-4'
CURE_TSR_DL5_PATH = './Datasets/CURE-TSR/Real_Train/DirtyLens-5'

CURE_TSR_GB1_PATH = './Datasets/CURE-TSR/Real_Train/GaussianBlur-1'
CURE_TSR_GB2_PATH = './Datasets/CURE-TSR/Real_Train/GaussianBlur-2'
CURE_TSR_GB3_PATH = './Datasets/CURE-TSR/Real_Train/GaussianBlur-3'
CURE_TSR_GB4_PATH = './Datasets/CURE-TSR/Real_Train/GaussianBlur-4'
CURE_TSR_GB5_PATH = './Datasets/CURE-TSR/Real_Train/GaussianBlur-5'

CURE_TSR_Noise1_PATH = './Datasets/CURE-TSR/Real_Train/Noise-1'
CURE_TSR_Noise2_PATH = './Datasets/CURE-TSR/Real_Train/Noise-2'
CURE_TSR_Noise3_PATH = './Datasets/CURE-TSR/Real_Train/Noise-3'
CURE_TSR_Noise4_PATH = './Datasets/CURE-TSR/Real_Train/Noise-4'
CURE_TSR_Noise5_PATH = './Datasets/CURE-TSR/Real_Train/Noise-5'

CURE_TSR_Rain1_PATH = './Datasets/CURE-TSR/Real_Train/Rain-1'
CURE_TSR_Rain2_PATH = './Datasets/CURE-TSR/Real_Train/Rain-2'
CURE_TSR_Rain3_PATH = './Datasets/CURE-TSR/Real_Train/Rain-3'
CURE_TSR_Rain4_PATH = './Datasets/CURE-TSR/Real_Train/Rain-4'
CURE_TSR_Rain5_PATH = './Datasets/CURE-TSR/Real_Train/Rain-5'

CURE_TSR_Snow1_PATH = './Datasets/CURE-TSR/Real_Train/Snow-1'
CURE_TSR_Snow2_PATH = './Datasets/CURE-TSR/Real_Train/Snow-2'
CURE_TSR_Snow3_PATH = './Datasets/CURE-TSR/Real_Train/Snow-3'
CURE_TSR_Snow4_PATH = './Datasets/CURE-TSR/Real_Train/Snow-4'
CURE_TSR_Snow5_PATH = './Datasets/CURE-TSR/Real_Train/Snow-5'

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

def level_chooser(sign_type, img_number, level):
    DL_file_name = f"01_{sign_type:02d}_05_{level:02d}_{img_number}.bmp"
    GB_file_name = f"01_{sign_type:02d}_07_{level:02d}_{img_number}.bmp"
    Noise_file_name = f"01_{sign_type:02d}_08_{level:02d}_{img_number}.bmp"
    Rain_file_name = f"01_{sign_type:02d}_09_{level:02d}_{img_number}.bmp"
    Snow_file_name = f"01_{sign_type:02d}_11_{level:02d}_{img_number}.bmp"


    if level == 1:
        DL_file_path = os.path.join(CURE_TSR_DL1_PATH, DL_file_name)
        GB_file_path = os.path.join(CURE_TSR_GB1_PATH, GB_file_name)
        Noise_file_path = os.path.join(CURE_TSR_Noise1_PATH, Noise_file_name)
        Rain_file_path = os.path.join(CURE_TSR_Rain1_PATH, Rain_file_name)
        Snow_file_path = os.path.join(CURE_TSR_Snow1_PATH, Snow_file_name)
    elif level == 2:
        DL_file_path = os.path.join(CURE_TSR_DL2_PATH, DL_file_name)
        GB_file_path = os.path.join(CURE_TSR_GB2_PATH, GB_file_name)
        Noise_file_path = os.path.join(CURE_TSR_Noise2_PATH, Noise_file_name)
        Rain_file_path = os.path.join(CURE_TSR_Rain2_PATH, Rain_file_name)
        Snow_file_path = os.path.join(CURE_TSR_Snow2_PATH, Snow_file_name)
    elif level == 3:
        DL_file_path = os.path.join(CURE_TSR_DL3_PATH, DL_file_name)
        GB_file_path = os.path.join(CURE_TSR_GB3_PATH, GB_file_name)
        Noise_file_path = os.path.join(CURE_TSR_Noise3_PATH, Noise_file_name)
        Rain_file_path = os.path.join(CURE_TSR_Rain3_PATH, Rain_file_name)
        Snow_file_path = os.path.join(CURE_TSR_Snow3_PATH, Snow_file_name)
    elif level == 4:
        DL_file_path = os.path.join(CURE_TSR_DL4_PATH, DL_file_name)
        GB_file_path = os.path.join(CURE_TSR_GB4_PATH, GB_file_name)
        Noise_file_path = os.path.join(CURE_TSR_Noise4_PATH, Noise_file_name)
        Rain_file_path = os.path.join(CURE_TSR_Rain4_PATH, Rain_file_name)
        Snow_file_path = os.path.join(CURE_TSR_Snow4_PATH, Snow_file_name)
    else:
        DL_file_path = os.path.join(CURE_TSR_DL5_PATH, DL_file_name)
        GB_file_path = os.path.join(CURE_TSR_GB5_PATH, GB_file_name)
        Noise_file_path = os.path.join(CURE_TSR_Noise5_PATH, Noise_file_name)
        Rain_file_path = os.path.join(CURE_TSR_Rain5_PATH, Rain_file_name)
        Snow_file_path = os.path.join(CURE_TSR_Snow5_PATH, Snow_file_name)

    return DL_file_path, GB_file_path, Noise_file_path, Rain_file_path, Snow_file_path


OUTPUT_PATH = './Outputs/CURE-TSR'

# chosen 1 image per signType for the ground truth
GT_list = ['0407', '0230', '0085', '0031', '0616', '0083', '0074', '0101', '0099', '0155', '0073', '0111', '0380', '0078']


def put_images(gt_image, noised_image, subfolder_path):
    denoised_bilateral = bilateral(noised_image)
    denoised_NLM10 = nlm_denoising(noised_image, 10)
    denoised_NLM20 = nlm_denoising(noised_image, 20)
    denoised_NLM30 = nlm_denoising(noised_image, 30)

    gt_image = Image.fromarray(np.uint8(gt_image))
    gt_image_path = os.path.join(subfolder_path, f"GT.png")

    noise_image_path = os.path.join(subfolder_path, f"Noised.png")
    noise_image = Image.fromarray(np.uint8(noised_image))

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


for i, idx in enumerate(GT_list):
    img_number = idx.zfill(4)
    sign_type = i+1

    GT_file_name = f"01_{sign_type:02d}_00_00_{img_number}.bmp"
    GT_file_path = os.path.join(CURE_TSR_GT_PATH, GT_file_name)
    for j in range(1, 6):
        DL_file_path, GB_file_path, Noise_file_path, Rain_file_path, Snow_file_path = level_chooser(sign_type, img_number, j)
        gt_img = mpimg.imread(GT_file_path)
        dl_img = mpimg.imread(DL_file_path)
        gb_img = mpimg.imread(GB_file_path)
        noise_img = mpimg.imread(Noise_file_path)
        rain_img = mpimg.imread(Rain_file_path)
        snow_img = mpimg.imread(Snow_file_path)

        dl_subpath = os.path.join(OUTPUT_PATH, f"{i}_dl_{j}")
        os.makedirs(dl_subpath, exist_ok=True)
        put_images(gt_img, dl_img, dl_subpath)

        gb_subpath = os.path.join(OUTPUT_PATH, f"{i}_gb_{j}")
        os.makedirs(gb_subpath, exist_ok=True)
        put_images(gt_img, gb_img, gb_subpath)

        noise_subpath = os.path.join(OUTPUT_PATH, f"{i}_noise_{j}")
        os.makedirs(noise_subpath, exist_ok=True)
        put_images(gt_img, noise_img, noise_subpath)

        rain_subpath = os.path.join(OUTPUT_PATH, f"{i}_rain_{j}")
        os.makedirs(rain_subpath, exist_ok=True)
        put_images(gt_img, rain_img, rain_subpath)

        snow_subpath = os.path.join(OUTPUT_PATH, f"{i}_snow_{j}")
        os.makedirs(snow_subpath, exist_ok=True)
        put_images(gt_img, snow_img, snow_subpath)






    # if os.path.exists(GT_file_path):
    #     # Load and display the image using matplotlib
    #     img = mpimg.imread(GT_file_path)
    #     plt.imshow(img)
    #     plt.title(f"Image: {GT_file_path}")
    #     plt.show()
    # else:
    #     print(f"File not found: {GT_file_path}")



