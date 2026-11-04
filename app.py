import streamlit as st
import cv2
import numpy as np
import time

# Custom Functions
def custom_ghosting(prev_frame, curr_frame):
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray1, gray2)
    _, binary_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    
    blended = cv2.addWeighted(prev_frame, 1.0, curr_frame, 0.7, 0)
    result = prev_frame.copy()
    cv2.copyTo(src=blended, mask=binary_mask, dst=result)
    return result

def custom_invisibility(bg_frame, curr_frame):
    bg_blur = cv2.GaussianBlur(bg_frame, (0, 0), sigmaX=3)
    img_blur = cv2.GaussianBlur(curr_frame, (0, 0), sigmaX=3)
    
    bg_gray = cv2.cvtColor(bg_blur, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(bg_gray, img_gray)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3)
    
    mask_normalized = mask.astype(float) / 255.0
    mask_3d = cv2.merge([mask_normalized, mask_normalized, mask_normalized])
    
    result = (curr_frame.astype(float) * (1.0 - mask_3d) + bg_frame.astype(float) * mask_3d).astype(np.uint8)
    return result


# OpenCV Built-in Functions
def opencv_ghosting(prev_frame, curr_frame, backSub):
    fg_mask = backSub.apply(curr_frame, learningRate=0.0)
    _, binary_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    blended = cv2.addWeighted(prev_frame, 1.0, curr_frame, 0.7, 0)
    result = prev_frame.copy()
    cv2.copyTo(src=blended, mask=binary_mask, dst=result)
    return result

def opencv_invisibility(curr_frame, backSub):
    fg_mask = backSub.apply(curr_frame, learningRate=0.01)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    bg_frame = backSub.getBackgroundImage()
    
    if bg_frame is None:
        return curr_frame
        
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
    fg_mask = cv2.GaussianBlur(fg_mask, (0, 0), sigmaX=3)
    
    mask_normalized = fg_mask.astype(float) / 255.0
    mask_3d = cv2.merge([mask_normalized, mask_normalized, mask_normalized])
    
    result = (curr_frame.astype(float) * (1.0 - mask_3d) + bg_frame.astype(float) * mask_3d).astype(np.uint8)
    return result


# Streamlit Application
st.set_page_config(layout="wide")
st.title("OpenCV Comparison: Custom Math vs MOG2 Built-ins")
st.sidebar.header("Controls")
effect_mode = st.sidebar.radio("Select Effect:", ("Invisibility Cloak", "Ghosting Trail"))
view_mode = st.sidebar.radio("View Mode:", ("Custom Implementation", "OpenCV MOG2", "Side-by-Side"))
run_stream = st.sidebar.checkbox("Start Video Stream", value=False)
capture_bg_btn = st.sidebar.button("Capture Static Background (Custom Only)")

frame_placeholder1 = st.empty()
frame_placeholder2 = st.empty()
if view_mode == "Side-by-Side":
    col1, col2 = st.columns(2)
    frame_placeholder1 = col1.empty()
    frame_placeholder2 = col2.empty()

def put_fps(img, fps, title):
    cv2.putText(img, f"{title} | FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img

if run_stream:
    phone_stream_url = "http://192.168.1.11:8080/video"  
    cap = cv2.VideoCapture(phone_stream_url)
    
    #cap = cv2.VideoCapture(0)
    
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    
    prev_frame = None
    custom_static_bg = None
    
    if "static_bg" not in st.session_state:
        st.session_state.static_bg = None
        
    if capture_bg_btn:
        ret, frame = cap.read()
        if ret:
            st.session_state.static_bg = frame

    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break
            
        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        if prev_frame is None:
            prev_frame = frame.copy()
        if st.session_state.static_bg is None:
            st.session_state.static_bg = frame.copy()

        custom_result = None
        opencv_result = None

        # --- Process Effects ---
        if effect_mode == "Ghosting Trail":
            if view_mode in ["Custom Implementation", "Side-by-Side"]:
                custom_result = custom_ghosting(prev_frame, frame)
            if view_mode in ["OpenCV MOG2", "Side-by-Side"]:
                opencv_result = opencv_ghosting(prev_frame, frame, backSub)
            
            prev_frame = cv2.addWeighted(prev_frame, 0.5, frame, 0.5, 0)
            
        elif effect_mode == "Invisibility Cloak":
            if view_mode in ["Custom Implementation", "Side-by-Side"]:
                custom_result = custom_invisibility(st.session_state.static_bg, frame)
            if view_mode in ["OpenCV MOG2", "Side-by-Side"]:
                opencv_result = opencv_invisibility(frame, backSub)

        # --- Display Results ---
        if view_mode == "Custom Implementation":
            custom_result_rgb = cv2.cvtColor(custom_result, cv2.COLOR_BGR2RGB)
            custom_result_rgb = put_fps(custom_result_rgb, fps, "Custom")
            frame_placeholder1.image(custom_result_rgb, channels="RGB", use_column_width=True)
            
        elif view_mode == "OpenCV MOG2":
            opencv_result_rgb = cv2.cvtColor(opencv_result, cv2.COLOR_BGR2RGB)
            opencv_result_rgb = put_fps(opencv_result_rgb, fps, "OpenCV MOG2")
            frame_placeholder1.image(opencv_result_rgb, channels="RGB", use_column_width=True)
            
        elif view_mode == "Side-by-Side":
            custom_result_rgb = cv2.cvtColor(custom_result, cv2.COLOR_BGR2RGB)
            custom_result_rgb = put_fps(custom_result_rgb, fps, "Custom")
            frame_placeholder1.image(custom_result_rgb, channels="RGB", use_column_width=True)
            
            if opencv_result is not None:
                opencv_result_rgb = cv2.cvtColor(opencv_result, cv2.COLOR_BGR2RGB)
                opencv_result_rgb = put_fps(opencv_result_rgb, fps, "OpenCV MOG2")
                frame_placeholder2.image(opencv_result_rgb, channels="RGB", use_column_width=True)

    cap.release()
else:
    st.info("Check 'Start Video Stream' in the sidebar to begin.")