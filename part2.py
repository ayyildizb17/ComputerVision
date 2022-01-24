import glob
import numpy as np
import os
import cv2
import moviepy.editor as mpy

def calculate_cdf(histogram):

    cdf = histogram.cumsum()

    normalized_cdf = cdf / float(cdf.max())

    return normalized_cdf


def create_lookup_table(source_cdf, target_cdf):

    lookup_table = np.zeros((256,1))
    lookup_val = 0
    for source_pixel_val in range(len(source_cdf)):
        lookup_val = lookup_val+1
        for target_pixel_val in range(255):
            if target_cdf[target_pixel_val] >= source_cdf[source_pixel_val]:
                lookup_val = target_pixel_val
                break
        lookup_table[source_pixel_val] = lookup_val
    return lookup_table


def matching_histograms(source_img,target_imgg):
    source_img_b,source_img_g, source_img_r = cv2.split(source_img)
    target_img_b,target_img_g,target_img_r = cv2.split(target_imgg)

    source_hist_b, bin_0 = np.histogram(source_img_b.flatten(), 256, [0, 256])
    source_hist_g, bin_1 = np.histogram(source_img_g.flatten(), 256, [0, 256])
    source_hist_r, bin_2 = np.histogram(source_img_r.flatten(), 256, [0, 256])
    target_hist_b, bin_3 = np.histogram(target_img_b.flatten(), 256, [0, 256])
    target_hist_g, bin_4 = np.histogram(target_img_g.flatten(), 256, [0, 256])
    target_hist_r, bin_5 = np.histogram(target_img_r.flatten(), 256, [0, 256])

    src_cdf_b = calculate_cdf(source_hist_b)
    src_cdf_g = calculate_cdf(source_hist_g)
    src_cdf_r = calculate_cdf(source_hist_r)
    trg_cdf_b = calculate_cdf(target_hist_b)
    trg_cdf_g = calculate_cdf(target_hist_g)
    trg_cdf_r = calculate_cdf(target_hist_r)

    blue_lookup_table = create_lookup_table(src_cdf_b, trg_cdf_b)
    green_lookup_table = create_lookup_table(src_cdf_g, trg_cdf_g)
    red_lookup_table = create_lookup_table(src_cdf_r, trg_cdf_r)

    blue_after_transform = cv2.LUT(source_img_b, blue_lookup_table)
    green_after_transform = cv2.LUT(source_img_g, green_lookup_table)
    red_after_transform = cv2.LUT(source_img_r, red_lookup_table)


    image_after_matching = cv2.merge([
        blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)

    return image_after_matching


background = cv2.imread('Malibu.jpg')
target_img = cv2.imread('target-img.jpg')

background_height = background.shape[0]
background_width = background.shape[1]
ratio = 360 / background_height

background = cv2.resize(background, (int(background_width * ratio), 360))

main_dir = 'cat'

images_list = []
cat_imgs = []
histo_index = []
images_path = os.path.join(main_dir, '*g')

for i in range(0, 180):
    cat_imgs.append(main_dir + "\\" + "cat_{}.png".format(i))

for images in cat_imgs:

    image = cv2.imread(images)
    matched_image = matching_histograms(image,target_img)


    flipped_img = np.fliplr(matched_image)

    foreground = np.logical_or(image[:, :, 1] < 180, image[:, :, 2] > 150)
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = image[nonzero_x, nonzero_y, :]

    new_frame = background.copy()
    new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values

    foreground = np.logical_or(flipped_img[:, :, 1] < 60, flipped_img[:, :, 2] < 50)

    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = flipped_img[nonzero_x, nonzero_y, :]
    new_frame[nonzero_x, (background.shape[1] - image.shape[1]) + nonzero_y, :] = nonzero_cat_values

    new_frame = new_frame[:, :, [2, 1, 0]]

    images_list.append(new_frame)


cv2.imshow('',images_list[178])
cv2.waitKey()

clip = mpy.ImageSequenceClip(images_list, fps=25)
audio = mpy.AudioFileClip('selfcontrol_part.wav').set_duration(clip.duration)
clip = clip.set_audio(audioclip=audio)
clip.write_videofile('part2_video.mp4', codec='libx264')