# digital-image-processing-cpp
cpp implementation for algorithms in the book "数字图像处理与机器视觉-Visual C++与Matlab实现"

To compile `CH3_pixel_operation.cpp`:

```sh
./compile.sh -DCH3 CH3_pixel_operation.cpp utility.cpp
```

In which `-DCH3` activates the `main` function in the source file, otherwise `main` function is ignored. And this source file includes `utility.h`, so we need to compile `CH3_pixel_operation.cpp` with `utility.h`'s implementation, which is `utility.cpp`.

Similar method for CH 4,5,6,7,8 to compile.

To compile `CH9_image_segmentation.cpp`:

```sh
./compile.sh -DCH9 CH9_image_segmentation.cpp CH3_pixel_operation.cpp CH5_spatial_domain_image_enhancement.cpp CH8_morphology_image_processing.cpp utility.cpp
```

Because it includes the corresponding headers.

## CH3 pixel operation

### Threshold
Set threshold as 100:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Threshold_100.png" alt="drawing" height="200"/>

### Linear Transform

Set `dFa` as `2.0`, set `dFb` as `-55`:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Linear_Transform_2.00_-55.00.png" alt="drawing" height="200"/>

### Gamma Transform

Set `gamma` as `1.8`, set `comp` as `0`:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Gamma_Transform_1.80_0.00.png" alt="drawing" height="200"/>

### Log Transform

Set `dC` as 10:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Log_Transform.png" alt="drawing" height="200"/>

### Partial Linear Transform

Set `x1`, `x2`, `y1`, `y2` as `20`, `50`, `100`, `200`:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Log_Paritial_Linear_Transform_20_50_100_200.png" alt="drawing" height="200"/>

### Histogram equalization

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Histogram_Equalization_128.png" alt="drawing" height="200"/>

### Histogram matching to dark 

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Histogram_Matching_to_Dark.png" alt="drawing" height="200"/>

### Histogram matching to light

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Histogram_Matching_to_Light.png" alt="drawing" height="200"/>

## CH4 geometric transformation

### Move

Move `20` right and `50` down:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Move_20_50_rgb.png" alt="drawing" height="200"/>

### Horizontal mirror

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Horizontal_Mirror_rgb.png" alt="drawing" height="200"/>

### Vertical mirror

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Vertical_Mirror_rgb.png" alt="drawing" height="200"/>

### Scale

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Scale_1.50_rgb.png" alt="drawing" height="200"/>

### Rotate

Rotate 30 degrees counterclockwise:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Rotate_30.00_rgb.png" alt="drawing" height="200"/>

### Image projection restore

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Projection_Restore_1.png" alt="drawing" height="200"/>

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Projection_Restore_2.png" alt="drawing" height="200"/>


## CH5 spatial domain image enhancement

### Smooth average

Kernel size 3:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/SmoothAvg.png" alt="drawing" height="200"/>

Kernel size 5:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/SmoothAvg5.png" alt="drawing" height="200"/>

Kernel size 7:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/SmoothAvg7.png" alt="drawing" height="200"/>

### Smooth gaussian

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/SmoothGauss.png" alt="drawing" height="200"/>

### Log edge detection

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/LogEdgeDetection.png" alt="drawing" height="200"/>

### Average vs Gaussian vs Median filter

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Average_vs_Gaussian_vs_Median.png" alt="drawing" height="200"/>

### Roberts cross gradient operator 

Positive 45 degrees v.s. Negative 45 degrees v.s. Positive+Negative 45 degrees:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Robert_P_vs_N_vs_P%2BN_45G.png" alt="drawing" height="200"/>

### Sobel gradient operator

Vertical v.s. Horizontal v.s. Vertical+Horizontal:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Sobel_V_vs_H_vs_V%2BH_G.png" alt="drawing" height="200"/>


### Laplacian operator

90 degrees rotation isotropy v.s. 45 degrees rotation isotropy v.s. Weighted

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Laplacian_90_vs_45_vs_Weighted_G.png" alt="drawing" height="200"/>

### Enhancement

Roberts positive 45 degrees:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/RobertP45G_vs_Enhanced_G.png" alt="drawing" height="200"/>

Sobel vertical:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/SobelVerticalG_vs_Enhanced_G.png" alt="drawing" height="200"/>

Laplacian weighted:

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Laplacian_Weighted_vs_Enhanced_G.png" alt="drawing" height="200"/>

## CH6 frequency domain image enhancement

### Ideal low pass filter
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Ideal_Low_pass_filter.png" alt="drawing" height="200"/>

### Gauss low pass filter
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Ideal_Low_pass_filter.png" alt="drawing" height="200"/>

### Gauss high pass filter
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Gauss_High_pass_filter.png" alt="drawing" height="200"/>

### Laplace filter
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Laplace_filter.png" alt="drawing" height="200"/>

### Gauss band rejection filter

Image and noised image in frequency domain:
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Image_and_noised_image_in_frequency_domain.png" alt="drawing" height="200"/>

Filter and filterd image in frequency domain:
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Filter_and_filted_image_in_frequency_domain.png" alt="drawing" height="200"/>

Before and after applying Gauss band rejection filter:
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Gauss_Band_Rejection_filter.png" alt="drawing" height="200"/>

## CH7 color image processing

### CMY
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/CMY_vs_CMY_BACK_RGB.png" alt="drawing" height="200"/>

### HSI
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/HSI_vs_HSI_BACK_RGB.png" alt="drawing" height="200"/>

### HSV
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/HSV_vs_HSV_BACK_RGB.png" alt="drawing" height="200"/>

### YUV
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/YUV_vs_YUV_BACK_RGB.png" alt="drawing" height="200"/>

### YIQ
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/YIQ_vs_YIQ_BACK_RGB.png" alt="drawing" height="200"/>

### Color compensating
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/plane_before_and_after_compensating.png" alt="drawing" height="200"/>

<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/rgb_before_and_after_compensating.png" alt="drawing" height="200"/>

### Color balancing
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/Before_and_after_color_balancing.png" alt="drawing" height="200"/>

## CH8 morphology image processing

### Erode using 3 x 3 square kernel and using cross kernel
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/erode_full_and_erode_cross.png" alt="drawing" height="200"/>

### Dilate using 3 x 3 square kernel and using cross kernel
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/dilate_full_and_dilate_cross.png" alt="drawing" height="200"/>

### Erode operation and open operation
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/erode_and_open.png" alt="drawing" height="200"/>

### Dilate operation and close operation
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/dilate_and_close.png" alt="drawing" height="200"/>

### Hit-or-miss transform with 50 x 50 square kernel
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/hit_or_miss_with_50x50_square.png" alt="drawing" height="200"/>

### Extract boundary
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/extract_boundary.png" alt="drawing" height="200"/>

### Trace boundary
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/trace_boundary.png" alt="drawing" height="200"/>

### Fill region
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/fill_region.png" alt="drawing" height="200"/>

### Label connected component
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/label_connected_component.png" alt="drawing" height="200"/>

### Thining
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/thining.png" alt="drawing" height="200"/>

### Pixelate
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/pixelate.png" alt="drawing" height="200"/>

### Convex hull unconstrained and constrained(using bounding rectangle)
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/convex_unconstrained_and_constrained.png" alt="drawing" height="200"/>

### Gray dilate
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/gray_dilated.png" alt="drawing" height="200"/>

### Gray erode
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/gray_eroded.png" alt="drawing" height="200"/>

### Gray opene
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/gray_opened.png" alt="drawing" height="200"/>

### Gray close
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/gray_closed.png" alt="drawing" height="200"/>

### Top-hat transform
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/top_hat.png" alt="drawing" height="200"/>

## CH9 image segmentation

### Edge detection using Prewitt operator
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/edge_Prewitt.png" alt="drawing" height="200"/>

### Edge detection using Sobel operator
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/edge_sobel.png" alt="drawing" height="200"/>

### Edge detection using LoG operator
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/edge_LoG.png" alt="drawing" height="200"/>

### Canny edge detector
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/edge_Canny.png" alt="drawing" height="200"/>

### Hough transformation for line detection
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/hough.png" alt="drawing" height="200"/>

### Automatically choose threshold for binarization
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/auto_threshold.png" alt="drawing" height="200"/>

### Region growing for image segmentation
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/region_grow.png" alt="drawing" height="200"/>

### Region splitting(decomposing) for image segmentation
<img src="https://github.com/keineahnung2345/digital-image-processing-cpp/blob/master/images/result/decompose.png" alt="drawing" height="200"/>
