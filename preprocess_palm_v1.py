import cv2
import mediapipe as mp
import numpy as np
import os


# extract ROI from palm vein images
# return: array of images, 128 x 128
def extract_ROI(imdir, roidir):
    drawing = mp.solutions.drawing_utils
    handsModule = mp.solutions.hands

    processed_roi = []

    with handsModule.Hands(static_image_mode=True) as hands:
        for im_path in os.listdir(imdir):
            image = cv2.imread(imdir + '/' + im_path)
            original = image
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            height, width, _ = image.shape

            if results.multi_hand_landmarks != None:
                for hand_landmarks in results.multi_hand_landmarks:
                    
                    # extract interested landmarks
                    norm_a = hand_landmarks.landmark[handsModule.HandLandmark.WRIST]
                    a = drawing._normalized_to_pixel_coordinates(norm_a.x, norm_a.y, width, height)

                    norm_b = hand_landmarks.landmark[handsModule.HandLandmark.THUMB_CMC]
                    b = drawing._normalized_to_pixel_coordinates(norm_b.x, norm_b.y, width, height)
                    
                    norm_e = hand_landmarks.landmark[handsModule.HandLandmark.INDEX_FINGER_MCP]
                    e = drawing._normalized_to_pixel_coordinates(norm_e.x, norm_e.y, width, height)
                    
                    norm_g = hand_landmarks.landmark[handsModule.HandLandmark.PINKY_MCP]
                    g = drawing._normalized_to_pixel_coordinates(norm_g.x, norm_g.y, width, height)
                    

                    # calculate keypoints based on extracted landmarks
                    if not (a and b  and e and g): continue
                    key_1 = (int((3 * a[0] - b[0]) / 2), int((3 * a[1] - b[1]) / 2))
                    key_2 = b # (int((b[0] + c[0]) / 2), int((b[1] + c[1]) / 2))
                    key_3 = e # (int((d[0] + e[0]) / 2), int((d[1] + e[1]) / 2))
                    key_4 = g # (int((f[0] + g[0]) / 2), int((f[1] + g[1]) / 2))
            

                    # warp images into rectangle
                    src_points = np.array([
                        key_1,
                        key_2,
                        key_3,
                        key_4
                    ], dtype='float32')

                    dst_points = np.array([
                        [0,0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype='float32')

                    mat = cv2.getPerspectiveTransform(src_points, dst_points)
                    image = cv2.warpPerspective(image, mat, (width, height))

                    image = cv2.resize(image, (128, 128))
                    processed_roi.append(image)
                    cv2.imwrite(roidir + im_path, image)
                    

                    # draw detected landmarks
                    # drawing.draw_landmarks(image, hand_landmarks, handsModule.HAND_CONNECTIONS)

    return np.array(processed_roi)