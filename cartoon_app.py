import cv2
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from scipy.interpolate import UnivariateSpline



def cartoonization(img, cartoon):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if cartoon == "Pencil Sketch":
        value = st.sidebar.slider('Tune the brightness of your sketch (the higher the value, the brighter your sketch)',
                                  0.0, 300.0, 250.0)
        kernel = st.sidebar.slider(
            'Tune the boldness of the edges of your sketch (the higher the value, the bolder the edges)', 1, 99, 25,
            step=2)

        gray_blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)

        cartoon = cv2.divide(gray, gray_blur, scale=value)

    if cartoon == "Detail Enhancement":
        smooth = st.sidebar.slider(
            'Tune the smoothness level of the image (the higher the value, the smoother the image)', 3, 99, 5, step=2)
        kernel = st.sidebar.slider('Tune the sharpness of the image (the lower the value, the sharper it is)', 1, 21, 3,
                                   step=2)
        edge_preserve = st.sidebar.slider(
            'Tune the color averaging effects (low: only similar colors will be smoothed, high: dissimilar color will be smoothed)',
            0.0, 1.0, 0.5)

        gray = cv2.medianBlur(gray, kernel)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)

        color = cv2.detailEnhance(img, sigma_s=smooth, sigma_r=edge_preserve)
        cartoon = cv2.bitwise_and(color, color, mask=edges)

    if cartoon == "Pencil Edges":
        kernel = st.sidebar.slider('Tune the sharpness of the sketch (the lower the value, the sharper it is)', 1, 99,
                                   25, step=2)
        laplacian_filter = st.sidebar.slider(
            'Tune the edge detection power (the higher the value, the more powerful it is)', 3, 9, 3, step=2)
        noise_reduction = st.sidebar.slider(
            'Tune the noise effects of your sketch (the higher the value, the noisier it is)', 10, 255, 150)

        gray = cv2.medianBlur(gray, kernel)
        edges = cv2.Laplacian(gray, -1, ksize=laplacian_filter)

        edges_inv = 255 - edges

        dummy, cartoon = cv2.threshold(edges_inv, noise_reduction, 255, cv2.THRESH_BINARY)

    if cartoon == "Bilateral Filter":
        smooth = st.sidebar.slider(
            'Tune the smoothness level of the image (the higher the value, the smoother the image)', 3, 99, 5, step=2)
        kernel = st.sidebar.slider('Tune the sharpness of the image (the lower the value, the sharper it is)', 1, 21, 3,
                                   step=2)
        edge_preserve = st.sidebar.slider(
            'Tune the color averaging effects (low: only similar colors will be smoothed, high: dissimilar color will be smoothed)',
            1, 100, 50)
        #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, kernel)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)

        color = cv2.bilateralFilter(img, smooth, edge_preserve, smooth)
        cartoon = cv2.bitwise_and(color, color, mask=edges)

    if cartoon == "Glitter Filter":

        kernel = st.sidebar.slider('Tune the sharpness of the image (the lower the value, the sharper it is)', 1, 21, 3,
                                   step=2)
        color_space = st.sidebar.slider(
            'Enter a value  for filter sigma in the color space '
            '(larger the value the farther colors within  pixel neighborhood will be mixed together.)', 3, 99, 5,
            step=2)
        coordinate_space = st.sidebar.slider(
                  'Enter a value to filter sigma in the coordinate space (A larger value means that farther pixels will influence each-other.)',
            0, 10, 1)

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_small = cv2.pyrDown(img)
        num_iter = 5
        for _ in range(num_iter):
            img_small = cv2.bilateralFilter(img_small, d=9, sigmaColor=color_space, sigmaSpace=coordinate_space)

        img_rgb = cv2.pyrUp(img_small)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, kernel)
        img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        array = cv2.bitwise_or(img, img_rgb)
        cartoon = cv2.bitwise_and(array, img_edge)

    if cartoon == "Canny Image":
        threshold_1 = st.sidebar.slider(
            'Tune the threshold level of the image (Lower the value higher the edges)', 1, 200, 3, step=2)
        threshold_2 = st.sidebar.slider('Tune the threshold level of the image (Higher the value lower the edges)', 1, 200, 3,
                                   step=2)
        cartoon = cv2.Canny(img, threshold_1, threshold_2)

    if cartoon == "Grayscale Image":

        cartoon = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if cartoon == "Grayscale-Blurred Image":
        kernel = st.sidebar.slider('Tune the sharpness of the image (the lower the value, the sharper it is)', 1, 21, 3,
                                   step=2)
        cartoon = cv2.medianBlur(gray, kernel)

    if cartoon == "Sepia-Filter":
        # sepia effect

            img_sepia = np.array(img, dtype=np.float64)  # converting to float to prevent loss
            img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                                            [0.349, 0.686, 0.168],
                                                            [0.393, 0.769,
                                                             0.189]]))  # multipying image with special sepia matrix
            img_sepia[np.where(img_sepia > 255)] = 255  # normalizing values greater than 255 to 255
            cartoon = np.array(img_sepia, dtype=np.uint8)

    if cartoon == "Pencil Sketch Effect: Colour":
        smooth = st.sidebar.slider(
            'Tune the smoothness level of the image (the higher the value, the smoother the image)', 3, 99, 5, step=2)

        edge_preserve = st.sidebar.slider(
            'Tune the color averaging effects (low: only similar colors will be smoothed, high: dissimilar color will be smoothed)',
            0.0, 1.0, 0.5)

        dummy,cartoon = cv2.pencilSketch(img, sigma_s=smooth, sigma_r=edge_preserve, shade_factor=0.1)

    if cartoon == "HDR Filter":
        smooth = st.sidebar.slider(
            'Tune the smoothness level of the image (the higher the value, the smoother the image)', 3, 99, 5, step=2)

        edge_preserve = st.sidebar.slider(
            'Tune the color averaging effects (low: only similar colors will be smoothed, high: dissimilar color will be smoothed)',
            0.0, 1.0, 0.5)
        cartoon = cv2.detailEnhance(img, sigma_s=smooth, sigma_r=edge_preserve)

    if cartoon == "Invert-Filter":
        cartoon = cv2.bitwise_not(img)

    if cartoon == "Summer Effect Filter":
        def LookupTable(x, y):
            spline = UnivariateSpline(x, y)
            return spline(range(256))

        increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel, red_channel = cv2.split(img)
        red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
        cartoon = cv2.merge((blue_channel, green_channel, red_channel))


    if cartoon == "Winter Effect Filter":
        def LookupTable(x, y):
            spline = UnivariateSpline(x, y)
            return spline(range(256))

        increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel, red_channel = cv2.split(img)
        red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
        cartoon = cv2.merge((blue_channel, green_channel, red_channel))

    if cartoon == "Sharp Effect":
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        cartoon = cv2.filter2D(img, -1, kernel)

    if cartoon == "Brightness Adjustment":
        adjustment = st.sidebar.slider(
            'Tune the brightness adjustment level of the image (the higher the value, the brighter the image)', -100, 100, 2, step=2)

        def bright(img, beta_value):
            img_bright = cv2.convertScaleAbs(img, beta=beta_value)
            return img_bright

        cartoon = bright(img, adjustment)


    return cartoon


st.write("""
          # Cartoonize Your Image!

          """
         )

st.write("This is an app to turn your photos into cartoon")

file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    img = np.array(image)

    option = st.sidebar.selectbox(
        'Which cartoon filters would you like to apply?',
        ('Pencil Sketch', 'Detail Enhancement', 'Pencil Edges', 'Bilateral Filter','Glitter Filter', 'Canny Image', 'Grayscale Image', 'Grayscale-Blurred Image','Sepia-Filter','Pencil Sketch Effect: Colour','HDR Filter','Invert-Filter','Summer Effect Filter','Winter Effect Filter','Sharp Effect','Brightness Adjustment'))

    st.text("Your original image")
    st.image(image, use_column_width=True)

    st.text("Your cartoonized image")
    cartoon = cartoonization(img, option)

    st.image(cartoon, use_column_width=True)