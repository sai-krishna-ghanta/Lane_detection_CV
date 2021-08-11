import numpy as np
import cv2
from camera_calibration import calib, undistort
from threshold import get_combined_gradients, get_combined_hls, combine_grad_hls
from line import Line, get_perspective_transform, get_lane_lines_img, illustrate_driving_lane, illustrate_info_panel, illustrate_driving_lane_with_topdownview
from moviepy.editor import VideoFileClip
import logging 
import streamlit as st
from PIL import Image

logging.basicConfig(filename='test.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')


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



def image_pipeline(img):
        img = np.array(Image.open(img))
        undist_img = undistort(img, mtx, dist)
        #if save_img == True:
            #cv2.imwrite('./output_images/01_undist_img.png', undist_img) 

        undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
        rows, cols = undist_img.shape[:2]
        ##if save_img == True:
            #cv2.imwrite('./output_images/02_resized_img.png', undist_img) 
    
        combined_gradient = get_combined_gradients(undist_img, th_sobelx, th_sobely, th_mag, th_dir)
        ##if save_img == True:
            #cv2.imwrite('./output_images/03_combined_gradient_img.png', combined_gradient) 

        combined_hls = get_combined_hls(undist_img, th_h, th_l, th_s)
        #if save_img == True:
            #cv2.imwrite('./output_images/04_combined_hls_img.png', combined_hls)
 
        combined_result = combine_grad_hls(combined_gradient, combined_hls)
        #if save_img == True:
            #cv2.imwrite('./output_images/05_combined_thresh_result_img.png', combined_result) 


        c_rows, c_cols = combined_result.shape[:2]
        s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
        s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

        src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
        dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

        warp_img, M, Minv = get_perspective_transform(combined_result, src, dst, (720, 720))
        #if save_img == True:
            #cv2.imwrite('./output_images/07_warped_img.png', warp_img) 

        searching_img = get_lane_lines_img(warp_img, left_line, right_line)
        #if save_img == True:
            #cv2.imwrite('./output_images/08_searching_img.png', searching_img) 

        w_comb_result, w_color_result = illustrate_driving_lane(searching_img, left_line, right_line)
        ##if save_img == True:
            #cv2.imwrite('./output_images/09_w_comb_result.png', w_comb_result) 
        ##if save_img == True:
            #cv2.imwrite('./output_images/10_w_color_result_img.png', w_color_result) 

        # Drawing the lines back down onto the road
        color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
        #if save_img == True:
            #cv2.imwrite('./output_images/11_color_result.png', color_result) 

        comb_result = np.zeros_like(undist_img)
        comb_result[220:rows - 12, 0:cols] = color_result
        #if save_img == True:
            #cv2.imwrite('./output_images/12_color_result_crop.png', color_result) 

        # Combine the result with the original image
        result = cv2.addWeighted(undist_img, 1, comb_result, 0.3, 0)
        
        return result
    # If working with video mode, use moviepy and process each frame and save the video.


html_temp = """
    <body style="background-color:red;">
    <div style="background-color:Teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Leaf Disease Detection</h2>
    </div>
    </body>
    """
st.markdown(html_temp, unsafe_allow_html=True)
file_image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
if file_image is not None:
    our_image = Image.open(file_image)
    st.success("File read -Hold On")
    result = image_pipeline(file_image)
    st.image(result)





