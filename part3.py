import numpy as np
import cv2
import cv
import moviepy.editor as moviepy

planes = np.zeros((9,472,4,3))
album_cover = cv2.imread('hotel-diablo.jpg')
cat_image = cv2. imread('Scat.jpg')

def getPerspective(sourcePoints, destinationPoints):

    a = np.zeros((8, 8))
    b = np.zeros((8))
    for i in range(4):
        a[i][0] = a[i+4][3] = sourcePoints[i][0]
        a[i][1] = a[i+4][4] = sourcePoints[i][1]
        a[i][2] = a[i+4][5] = 1
        a[i][3] = a[i][4] = a[i][5] = 0
        a[i+4][0] = a[i+4][1] = a[i+4][2] = 0
        a[i][6] = -sourcePoints[i][0]*destinationPoints[i][0]
        a[i][7] = -sourcePoints[i][1]*destinationPoints[i][0]
        a[i+4][6] = -sourcePoints[i][0]*destinationPoints[i][1]
        a[i+4][7] = -sourcePoints[i][1]*destinationPoints[i][1]
        b[i] = destinationPoints[i][0]
        b[i+4] = destinationPoints[i][1]

    x = np.linalg.lstsq(a, b, rcond=None)[0]

    x.resize((9,), refcheck=False)
    x[8] = 1
    return x.reshape((3,3))


for i in range(1,10):
	with open("Plane_"+str(i)+".txt") as f:
		content = f.readlines()
		for line_id in range(len(content)):
			sel_line = content[line_id]
			sel_line = sel_line.replace(')\n', '').replace("(", '').split(")")

			for point_id in range(4):
				sel_point = sel_line[point_id].split(" ")

				planes[i-1,line_id,point_id,0] = float(sel_point[0])
				planes[i-1,line_id,point_id,1] = float(sel_point[1])
				planes[i-1,line_id,point_id,2] = float(sel_point[2])
				area = planes[i-1].shape[0]*planes[i-1].shape[1]

images_list = []
dest_list = []
album_cover_h = album_cover.shape[0]
album_cover_w = album_cover.shape[1]
pts_album = np.float32([[album_cover_w,0],[0,0],[0,album_cover_h],[album_cover_w,album_cover_h]])
for i in range(472):
	blank_image = np.ones((322,572,3), np.uint8)*255
	blank_image.fill(255)
	for j in range(9):

			pts = planes[j,i,:,:].squeeze()[:,0:2].astype(np.float32)

			temp = np.copy(pts[3,:])
			pts[3, :] = pts[2,:]
			pts[2, :] = temp

			pts = pts.reshape((-1, 1, 2))

			dest = pts.squeeze()[:,0:2]
			src = pts_album

			perspectiveTransform = getPerspective(src, dest)
			img = cv2.warpPerspective(album_cover, perspectiveTransform, (572, 322))

			foreground = np.logical_or((np.logical_or(img[:, :, 1] < 256, img[:, :, 2] < 256)),img[:,:,0]<256)

			nonzero_x, nonzero_y = np.nonzero(foreground)

			nonzero_img_values = img[nonzero_x, nonzero_y, :]

			new_frame = blank_image.copy()
			new_frame[nonzero_x, nonzero_y, :] = nonzero_img_values

			new_frame = new_frame[:, :, [2, 1, 0]]
			cv2.imshow('',new_frame)
			cv2.waitKey()

			images_list.append(new_frame)

clip = moviepy.ImageSequenceClip(images_list, fps = 7)
clip.write_videofile("part3_vid.mp4", codec="libx264")





