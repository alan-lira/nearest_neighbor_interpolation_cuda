/*
MIT License

Copyright (c) 2022 Alan Lira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

void string_to_lower_case(char *string) {
    for (int i = 0; string[i]; i++) {
        string[i] = tolower(string[i]);
    }
}

void string_to_upper_case(char *string) {
    for (int i = 0; string[i]; i++) {
        string[i] = toupper(string[i]);
    }
}

void nearest_neighbor_interpolation(unsigned char *input_image,
                                    unsigned char *output_image,
                                    int input_image_width,
                                    int output_image_width,
                                    int output_image_height,
                                    float scale_x,
                                    float scale_y,
                                    int image_channels) {
    for (int y = 0; y < output_image_height; y++) {
        for (int x = 0; x < output_image_width; x++) {
            int x_nearest = (int) x / scale_x;
            int y_nearest = (int) y / scale_y;
            for (int c = 0; c < image_channels; c++) {
                int index_input = (input_image_width * y_nearest * image_channels) + (x_nearest * image_channels);
                int index_output = (output_image_width * y * image_channels) + (x * image_channels);
                output_image[index_output + c] = input_image[index_input + c];
            }
        }
    }
}

void execute_on_cpu(unsigned char *input_image,
                    unsigned char *output_image,
                    int input_image_width,
                    int output_image_width,
                    int output_image_height,
                    float scale_x,
                    float scale_y,
                    int image_channels) {
    // Execute the Nearest Neighbor Interpolation (Image Scaling) on the Host (CPU).
    nearest_neighbor_interpolation(input_image,
                                   output_image,
                                   input_image_width,
                                   output_image_width,
                                   output_image_height,
                                   scale_x,
                                   scale_y,
                                   image_channels);
}

// Wrapper for CUDA Functions Calls.
#define CUDA_CHECK(call) \
    if ((call) != cudaSuccess) { \
        cudaError_t cuda_error = cudaGetLastError(); \
	printf("The Following CUDA Error Occurred: %s.\n", cudaGetErrorString(cuda_error)); \
        exit(4); \
    }

__global__ void nearest_neighbor_interpolation_kernel(unsigned char *input_image,
                                                      unsigned char *output_image,
                                                      int input_image_width,
                                                      int output_image_width,
                                                      int output_image_height,
                                                      float scale_x,
                                                      float scale_y,
                                                      int image_channels) {
    int y = (blockDim.y * blockIdx.y) + threadIdx.y;
    int x = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (y < output_image_height && x < output_image_width) {
        int x_nearest = (int) x / scale_x;
        int y_nearest = (int) y / scale_y;
        for (int c = 0; c < image_channels; c++) {
            int index_input = (input_image_width * y_nearest * image_channels) + (x_nearest * image_channels);
            int index_output = (output_image_width * y * image_channels) + (x * image_channels);
            output_image[index_output + c] = input_image[index_input + c];
        }
    }
}

void execute_on_gpu_with_cuda(unsigned char *input_image_host,
                              unsigned char *output_image_host,
                              int input_image_width,
                              int input_image_height,
                              int input_image_channels,
                              int output_image_width,
                              int output_image_height,
                              int output_image_channels,
                              float scale_x,
                              float scale_y) {
    // Input Image's Data Memory Alloc for the Device (GPU).
    unsigned char *input_image_device;
    CUDA_CHECK(cudaMalloc(&input_image_device,
                          sizeof(unsigned char) * input_image_width * input_image_height * input_image_channels));
    // Copy the Input Image's Data From Host (CPU) to Device (GPU).
    CUDA_CHECK(cudaMemcpy(input_image_device,
                          input_image_host,
                          sizeof(unsigned char) * input_image_width * input_image_height * input_image_channels,
                          cudaMemcpyHostToDevice));
    // Output Image's Data Memory Alloc for the Device (GPU).
    unsigned char *output_image_device;
    CUDA_CHECK(cudaMalloc(&output_image_device,
                          sizeof(unsigned char) * output_image_width * output_image_height * output_image_channels));
    // Set the Number of Threads per Block (dimBlock) and the Number of Blocks (dimGrid).
    dim3 dimBlock(32, 32);
    dim3 dimGrid(ceil((float) output_image_width / dimBlock.x), ceil((float) output_image_height / dimBlock.y));
    // Execute the Nearest Neighbor Interpolation Kernel (Image Scaling) on the Device (GPU).
    nearest_neighbor_interpolation_kernel<<<dimGrid, dimBlock>>>(input_image_device,
                                                                 output_image_device,
                                                                 input_image_width,
                                                                 output_image_width,
                                                                 output_image_height,
                                                                 scale_x,
                                                                 scale_y,
                                                                 input_image_channels);
    // Wait for the GPU to Finish the Kernel Execution.
    cudaDeviceSynchronize();
    // Copy the Output Image's Data From Device (GPU) to Host (CPU).
    CUDA_CHECK(cudaMemcpy(output_image_host,
                          output_image_device,
                          sizeof(unsigned char) * output_image_width * output_image_height * output_image_channels,
                          cudaMemcpyDeviceToHost));
    // Free the Memory Allocated for the Device (GPU).
    CUDA_CHECK(cudaFree(input_image_device));
    CUDA_CHECK(cudaFree(output_image_device));
}

int main(int argc, char **argv) {
    // Begin.
    // Check the Number of Arguments Provided (Expected: argc = 7).
    if (argc != 7) {
        printf("USAGE: transform_image <CPU | GPU_CUDA> <Input_Image_Path> <Output_Image_Width> <Output_Image_Height> <Output_Image_Channels> <Output_Image_Path>\n");
	exit(1);
    }
    // Parse the Arguments Provided (argv).
    char *execution_type = argv[1];
    char *input_image_file = argv[2];
    int output_image_width = atoi(argv[3]);
    int output_image_height = atoi(argv[4]);
    int output_image_channels = atoi(argv[5]);
    char *output_image_file = argv[6];
    // Initialize the Input Image's Variables (Width, Height, and Channels).
    // Channels = 1 --> Grey
    // Channels = 2 --> Grey, Alpha
    // Channels = 3 --> Red, Green, Blue
    // Channels = 4 --> Red, Green, Blue, Alpha
    int input_image_width = 0, input_image_height = 0, input_image_channels = 0;
    // Load the Input Image Using stb_image and Return the Resulting Data and Its Properties:
    // Width, Height, and Number of 8-Bit Components (Channels) per Pixel in the Image.
    int input_image_desired_channels = 0; // Default Value: 0 (All).
    unsigned char *input_image_host = stbi_load(input_image_file,
                                                &input_image_width,
                                                &input_image_height,
                                                &input_image_channels,
                                                input_image_desired_channels);
    // Check if the Input Image Was Succesfully Loaded on the Host (CPU).
    if (input_image_host == NULL) {
        printf("ERROR When Trying to Load the Input Image '%s'!\n", input_image_file);
        exit(2);
    }
    // Check if the Number of Channels on Input and Output Images Are Equal.
    if (input_image_channels != output_image_channels) {
        printf("The Number of Channels on the Input and Output Images Must be Equal! (Input Image's Channels: %d)\n", input_image_channels);
        exit(3);
    }
    // Input Image's Summary.
    printf("Input Image Loaded: '%s' --> Width = %dpx, Height = %dpx, and Channels = %d.\n",
           input_image_file,
           input_image_width,
           input_image_height,
           input_image_channels);
    // Output Image's Data Memory Alloc for the Host (CPU).
    unsigned char *output_image_host;
    output_image_host = (unsigned char *) malloc(sizeof(unsigned char) * output_image_width * output_image_height * output_image_channels);
    // Calculate the Scaling Factors (scale_x and scale_y).
    // A Scale Factor < 1 Indicates Image Shrinking;
    // A Scale Factor > 1 Indicates Image Stretching.
    float scale_x = (float) output_image_width / input_image_width;
    float scale_y = (float) output_image_height / input_image_height;
    // Set the Processing Time Variables.
    clock_t t = clock();
    double runtime_in_seconds = 0.0;
    // Get the Execution Type String and Transform It (Lower Case).
    string_to_lower_case(execution_type);
    // Process the Input Image (Using CPU Only or Using GPU With CUDA).
    if (strcmp(execution_type, "cpu") == 0) {
        printf("Processing the Input Image '%s' Using CPU Only (Scaling Factor: scale_x = %.2f, scale_y = %.2f)...\n",
               input_image_file,
               scale_x,
               scale_y);
        execute_on_cpu(input_image_host,
                       output_image_host,
                       input_image_width,
                       output_image_width,
                       output_image_height,
                       scale_x,
                       scale_y,
                       input_image_channels);
    } else if (strcmp(execution_type, "gpu_cuda") == 0) {
        printf("Processing the Input Image '%s' Using GPU With CUDA (Scaling Factor: scale_x = %.2f, scale_y = %.2f)...\n",
               input_image_file,
               scale_x,
               scale_y);
        execute_on_gpu_with_cuda(input_image_host,
                                 output_image_host,
                                 input_image_width,
                                 input_image_height,
                                 input_image_channels,
                                 output_image_width,
                                 output_image_height,
                                 output_image_channels,
                                 scale_x,
                                 scale_y);
    }
    // Calculate the Input Image Processing Time.
    t = clock() - t;
    runtime_in_seconds = (double) t / CLOCKS_PER_SEC;
    // Get the Execution Type String and Transform It (Upper Case).
    string_to_upper_case(execution_type);
    // Print the Input Image Processing Time.
    printf("Input Image Processing Time Using %s: %.2f ms (%.2f s)\n",
           execution_type,
           (runtime_in_seconds * 1000),
           runtime_in_seconds);
    // Write the Output Image JPG File Using stb_image.
    int output_image_quality = 100;
    stbi_write_jpg(output_image_file,
                   output_image_width,
                   output_image_height,
                   output_image_channels,
                   output_image_host,
                   output_image_quality);
    // Output Image's Summary.
    printf("Output Image Saved: '%s' --> Width = %dpx, Height = %dpx, Channels = %d, and Quality = %d%%.\n",
           output_image_file,
           output_image_width,
           output_image_height,
           output_image_channels,
           output_image_quality);
    // Free the Memory Allocated for the Host (CPU).
    stbi_image_free(input_image_host);
    free(output_image_host);
    // End.
    exit(0);
}

