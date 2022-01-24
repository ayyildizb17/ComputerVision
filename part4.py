
import cv2



album_cover = cv2.imread('hotel-diablo.jpg')

album_cover_h = album_cover.shape[0]
album_cover_w = album_cover.shape[1]

pts_center = tuple([album_cover_w/2,album_cover_h/2])
pts_top_left = tuple([0,0])
angle = 60
rot_mat = cv2.getRotationMatrix2D(pts_center, angle, 1.0)
rotated_img_center = cv2.warpAffine(album_cover, rot_mat, album_cover.shape[1::-1], flags=cv2.INTER_LINEAR)

rot_mat2 = cv2.getRotationMatrix2D(pts_top_left, 360-angle, 1.0)
rotated_img_tleft = cv2.warpAffine(album_cover, rot_mat2, album_cover.shape[1::-1], flags=cv2.INTER_LINEAR)

center = 'rotated-center.jpg'
top_left = 'rotated-top-left.jpg'
cv2.imwrite(center,rotated_img_center)
cv2.imwrite(top_left,rotated_img_tleft)
