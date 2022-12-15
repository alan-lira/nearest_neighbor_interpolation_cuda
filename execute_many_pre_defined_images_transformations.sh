#!/bin/bash

# MIT License

# Copyright (c) 2022 Alan Lira

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Script Begin.

input_images_folder="input_images"
output_images_folder="output_images"

# Create Output Images' Folder, If Not Exists (Otherwise Delete It's Content).
mkdir -p $output_images_folder && rm -rf $output_images_folder/*

# Execute Image Transformations Using CPU and GPU With Cuda.

# ------------
# 4:3 Images
# ------------

# [640 x 480 Pixels]

# x_scale = 0.25, y_scale = 0.25:
./transform_image cpu $input_images_folder/cat_640_480.jpg 160 120 3 $output_images_folder/cat_160_120.jpg
./transform_image gpu_cuda $input_images_folder/cat_640_480.jpg 160 120 3 $output_images_folder/cat_160_120.jpg

# x_scale = 0.5, y_scale = 0.5:
./transform_image cpu $input_images_folder/cat_640_480.jpg 320 240 3 $output_images_folder/cat_320_240.jpg
./transform_image gpu_cuda $input_images_folder/cat_640_480.jpg 320 240 3 $output_images_folder/cat_320_240.jpg

# x_scale = 2, y_scale = 2:
./transform_image cpu $input_images_folder/cat_640_480.jpg 1280 960 3 $output_images_folder/cat_1280_960.jpg
./transform_image gpu_cuda $input_images_folder/cat_640_480.jpg 1280 960 3 $output_images_folder/cat_1280_960.jpg

# x_scale = 5, y_scale = 5:
./transform_image cpu $input_images_folder/cat_640_480.jpg 3200 2400 3 $output_images_folder/cat_3200_2400.jpg
./transform_image gpu_cuda $input_images_folder/cat_640_480.jpg 3200 2400 3 $output_images_folder/cat_3200_2400.jpg

# x_scale = 10, y_scale = 10:
./transform_image cpu $input_images_folder/cat_640_480.jpg 6400 4800 3 $output_images_folder/cat_6400_4800.jpg
./transform_image gpu_cuda $input_images_folder/cat_640_480.jpg 6400 4800 3 $output_images_folder/cat_6400_4800.jpg

# [800 x 600 Pixels]

# x_scale = 0.25, y_scale = 0.25:
./transform_image cpu $input_images_folder/cat_800_600.jpg 200 150 3 $output_images_folder/cat_200_150.jpg
./transform_image gpu_cuda $input_images_folder/cat_800_600.jpg 200 150 3 $output_images_folder/cat_200_150.jpg

# x_scale = 0.5, y_scale = 0.5:
./transform_image cpu $input_images_folder/cat_800_600.jpg 400 300 3 $output_images_folder/cat_400_300.jpg
./transform_image gpu_cuda $input_images_folder/cat_800_600.jpg 400 300 3 $output_images_folder/cat_400_300.jpg

# x_scale = 2, y_scale = 2:
./transform_image cpu $input_images_folder/cat_800_600.jpg 1600 1200 3 $output_images_folder/cat_1600_1200.jpg
./transform_image gpu_cuda $input_images_folder/cat_800_600.jpg 1600 1200 3 $output_images_folder/cat_1600_1200.jpg

# x_scale = 5, y_scale = 5:
./transform_image cpu $input_images_folder/cat_800_600.jpg 4000 3000 3 $output_images_folder/cat_4000_3000.jpg
./transform_image gpu_cuda $input_images_folder/cat_800_600.jpg 4000 3000 3 $output_images_folder/cat_4000_3000.jpg

# x_scale = 10, y_scale = 10:
./transform_image cpu $input_images_folder/cat_800_600.jpg 8000 6000 3 $output_images_folder/cat_8000_6000.jpg
./transform_image gpu_cuda $input_images_folder/cat_800_600.jpg 8000 6000 3 $output_images_folder/cat_8000_6000.jpg

# [1024 x 768 Pixels]

# x_scale = 0.25, y_scale = 0.25:
./transform_image cpu $input_images_folder/cat_1024_768.jpg 256 192 3 $output_images_folder/cat_256_192.jpg
./transform_image gpu_cuda $input_images_folder/cat_1024_768.jpg 256 192 3 $output_images_folder/cat_256_192.jpg

# x_scale = 0.5, y_scale = 0.5:
./transform_image cpu $input_images_folder/cat_1024_768.jpg 512 384 3 $output_images_folder/cat_512_384.jpg
./transform_image gpu_cuda $input_images_folder/cat_1024_768.jpg 512 384 3 $output_images_folder/cat_512_384.jpg

# x_scale = 2, y_scale = 2:
./transform_image cpu $input_images_folder/cat_1024_768.jpg 2048 1536 3 $output_images_folder/cat_2048_1536.jpg
./transform_image gpu_cuda $input_images_folder/cat_1024_768.jpg 2048 1536 3 $output_images_folder/cat_2048_1536.jpg

# x_scale = 5, y_scale = 5:
./transform_image cpu $input_images_folder/cat_1024_768.jpg 5120 3840 3 $output_images_folder/cat_5120_3840.jpg
./transform_image gpu_cuda $input_images_folder/cat_1024_768.jpg 5120 3840 3 $output_images_folder/cat_5120_3840.jpg

# x_scale = 10, y_scale = 10:
./transform_image cpu $input_images_folder/cat_1024_768.jpg 10240 7680 3 $output_images_folder/cat_10240_7680.jpg
./transform_image gpu_cuda $input_images_folder/cat_1024_768.jpg 10240 7680 3 $output_images_folder/cat_10240_7680.jpg

# ------------
# 16:9 Images
# ------------

# [1280 x 720 Pixels]

# x_scale = 0.25, y_scale = 0.25:
./transform_image cpu $input_images_folder/cat_1280_720.jpg 320 180 3 $output_images_folder/cat_320_180.jpg
./transform_image gpu_cuda $input_images_folder/cat_1280_720.jpg 320 180 3 $output_images_folder/cat_320_180.jpg

# x_scale = 0.5, y_scale = 0.5:
./transform_image cpu $input_images_folder/cat_1280_720.jpg 640 360 3 $output_images_folder/cat_640_360.jpg
./transform_image gpu_cuda $input_images_folder/cat_1280_720.jpg 640 360 3 $output_images_folder/cat_640_360.jpg

# x_scale = 2, y_scale = 2:
./transform_image cpu $input_images_folder/cat_1280_720.jpg 2560 1440 3 $output_images_folder/cat_2560_1440.jpg
./transform_image gpu_cuda $input_images_folder/cat_1280_720.jpg 2560 1440 3 $output_images_folder/cat_2560_1440.jpg

# x_scale = 5, y_scale = 5:
./transform_image cpu $input_images_folder/cat_1280_720.jpg 6400 3600 3 $output_images_folder/cat_6400_3600.jpg
./transform_image gpu_cuda $input_images_folder/cat_1280_720.jpg 6400 3600 3 $output_images_folder/cat_6400_3600.jpg

# x_scale = 10, y_scale = 10:
./transform_image cpu $input_images_folder/cat_1280_720.jpg 12800 7200 3 $output_images_folder/cat_12800_7200.jpg
./transform_image gpu_cuda $input_images_folder/cat_1280_720.jpg 12800 7200 3 $output_images_folder/cat_12800_7200.jpg

# [1920 x 1080 Pixels]

# x_scale = 0.25, y_scale = 0.25:
./transform_image cpu $input_images_folder/cat_1920_1080.jpg 480 270 3 $output_images_folder/cat_480_270.jpg
./transform_image gpu_cuda $input_images_folder/cat_1920_1080.jpg 480 270 3 $output_images_folder/cat_480_270.jpg

# x_scale = 0.5, y_scale = 0.5:
./transform_image cpu $input_images_folder/cat_1920_1080.jpg 960 540 3 $output_images_folder/cat_960_540.jpg
./transform_image gpu_cuda $input_images_folder/cat_1920_1080.jpg 960 540 3 $output_images_folder/cat_960_540.jpg

# x_scale = 2, y_scale = 2:
./transform_image cpu $input_images_folder/cat_1920_1080.jpg 3840 2160 3 $output_images_folder/cat_3840_2160.jpg
./transform_image gpu_cuda $input_images_folder/cat_1920_1080.jpg 3840 2160 3 $output_images_folder/cat_3840_2160.jpg

# x_scale = 5, y_scale = 5:
./transform_image cpu $input_images_folder/cat_1920_1080.jpg 9600 5400 3 $output_images_folder/cat_9600_5400.jpg
./transform_image gpu_cuda $input_images_folder/cat_1920_1080.jpg 9600 5400 3 $output_images_folder/cat_9600_5400.jpg

# x_scale = 10, y_scale = 10:
./transform_image cpu $input_images_folder/cat_1920_1080.jpg 19200 10800 3 $output_images_folder/cat_19200_10800.jpg
./transform_image gpu_cuda $input_images_folder/cat_1920_1080.jpg 19200 10800 3 $output_images_folder/cat_19200_10800.jpg

# [3840 x 2160 Pixels]

# x_scale = 0.25, y_scale = 0.25:
./transform_image cpu $input_images_folder/cat_3840_2160.jpg 960 540 3 $output_images_folder/cat_960_540.jpg
./transform_image gpu_cuda $input_images_folder/cat_3840_2160.jpg 960 540 3 $output_images_folder/cat_960_540.jpg

# x_scale = 0.5, y_scale = 0.5:
./transform_image cpu $input_images_folder/cat_3840_2160.jpg 1920 1080 3 $output_images_folder/cat_1920_1080.jpg
./transform_image gpu_cuda $input_images_folder/cat_3840_2160.jpg 1920 1080 3 $output_images_folder/cat_1920_1080.jpg

# x_scale = 2, y_scale = 2:
./transform_image cpu $input_images_folder/cat_3840_2160.jpg 7680 4320 3 $output_images_folder/cat_7680_4320.jpg
./transform_image gpu_cuda $input_images_folder/cat_3840_2160.jpg 7680 4320 3 $output_images_folder/cat_7680_4320.jpg

# x_scale = 5, y_scale = 5:
./transform_image cpu $input_images_folder/cat_3840_2160.jpg 19200 10800 3 $output_images_folder/cat_19200_10800.jpg
./transform_image gpu_cuda $input_images_folder/cat_3840_2160.jpg 19200 10800 3 $output_images_folder/cat_19200_10800.jpg

# x_scale = 10, y_scale = 10:
# Currently Not Working:
# (CPU) Segmentation fault (core dumped)
# (GPU) The Following CUDA Error Occurred: an illegal memory access was encountered.
# ./transform_image cpu $input_images_folder/cat_3840_2160.jpg 38400 21600 3 $output_images_folder/cat_38400_21600.jpg
# ./transform_image gpu_cuda $input_images_folder/cat_3840_2160.jpg 38400 21600 3 $output_images_folder/cat_38400_21600.jpg

# Script End.
exit 0

