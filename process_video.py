import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from camera_calibration import calib, undistort
from threshold import get_combined_gradients, get_combined_hls, combine_grad_hls
from line import Line, get_perspective_transform, get_lane_lines_img, illustrate_driving_lane, illustrate_info_panel, illustrate_driving_lane_with_topdownview
from moviepy.editor import VideoFileClip
import logging 

logging.basicConfig(filename='test.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#       Select desired input name/type          #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#input_type = "image"
#input_type = 'image'
input_type = 'video' 
#input_type = 'frame_by_frame'
#input_name = 'test_images/calibration1.jpg'
input_name = 'project_video.mp4' 
#input_name = 'challenge_video.mp4'
#input_name = 'harder_challenge_video.mp4'

# If input_type is `image`, select whether you'd like to save intermediate images or not. 
save_img = True

left_line = Line()
right_line = Line()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#   Tune Parameters for different inputs        #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)

# camera matrix & distortion coefficient
mtx, dist = calib()


def pipeline(frame):
    # Correcting for Distortion

    undist_img = undistort(frame, mtx, dist)
    # resize video
    undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
    rows, cols = undist_img.shape[:2]

    combined_gradient = get_combined_gradients(undist_img, th_sobelx, th_sobely, th_mag, th_dir)

    combined_hls = get_combined_hls(undist_img, th_h, th_l, th_s)

    combined_result = combine_grad_hls(combined_gradient, combined_hls)

    c_rows, c_cols = combined_result.shape[:2]
    s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
    s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

    src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
    dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

    warp_img, M, Minv = get_perspective_transform(combined_result, src, dst, (720, 720))

    searching_img = get_lane_lines_img(warp_img, left_line, right_line)

    w_comb_result, w_color_result = illustrate_driving_lane(searching_img, left_line, right_line)

    # Drawing the lines back down onto the road
    color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
    lane_color = np.zeros_like(undist_img)
    lane_color[220:rows - 12, 0:cols] = color_result

    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, lane_color, 0.3, 0)

    info_panel, birdeye_view_panel = np.zeros_like(result),  np.zeros_like(result)
    info_panel[5:110, 5:325] = (255, 255, 255)
    birdeye_view_panel[5:110, cols-111:cols-6] = (255, 255, 255)
    
    info_panel = cv2.addWeighted(result, 1, info_panel, 0.2, 0)
    birdeye_view_panel = cv2.addWeighted(info_panel, 1, birdeye_view_panel, 0.2, 0)
    road_map = illustrate_driving_lane_with_topdownview(w_color_result, left_line, right_line)
    birdeye_view_panel[10:105, cols-106:cols-11] = road_map
    birdeye_view_panel = illustrate_info_panel(birdeye_view_panel, left_line, right_line)
    
    return birdeye_view_panel 

cap = cv2.VideoCapture(input_name)
if (cap.isOpened()== False):
    print("Error opening video stream or file")
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = pipeline(frame)
    if ret == True:
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()



file_video = input_name
if input_type == 'video':
    white_output = "./output_videos/video_out_1.mp4"
    frame = VideoFileClip(file_video)
    white_clip = frame.fl_image(pipeline)
    #white_clip.write_videofile(white_output, audio=False)
    logging.debug("This file is ran and outputs are generated",file_video)