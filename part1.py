
import numpy as np
import os
import cv2
import moviepy.editor as mpy

background = cv2.imread('Malibu.jpg')

background_height = background.shape[0]
background_width = background.shape[1]
ratio = 360 / background_height

background = cv2.resize(background, (int(background_width * ratio), 360))

main_dir = 'cat'

images_list = []
cat_imgs = []

images_path = os.path.join(main_dir, '*g')

for i in range(0, 180):
    cat_imgs.append(main_dir + "\\" + "cat_{}.png".format(i))

for images in cat_imgs:
    image = cv2.imread(images)
    flipped_img = np.fliplr(image)

    foreground = np.logical_or(image[:, :, 1] < 180, image[:, :, 2] > 150)
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = image[nonzero_x, nonzero_y, :]

    new_frame = background.copy()
    new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values

    foreground = np.logical_or(flipped_img[:, :, 1] < 180, flipped_img[:, :, 2] > 150)
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = flipped_img[nonzero_x, nonzero_y, :]
    new_frame[nonzero_x, (background.shape[1] - image.shape[1]) + nonzero_y, :] = nonzero_cat_values

    new_frame = new_frame[:, :, [2, 1, 0]]

    images_list.append(new_frame)


clip = mpy.ImageSequenceClip(images_list, fps=25)
audio = mpy.AudioFileClip('selfcontrol_part.wav').set_duration(clip.duration)
clip = clip.set_audio(audioclip=audio)
clip.write_videofile('part1_video.mp4', codec='libx264')