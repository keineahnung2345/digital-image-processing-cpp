# digital-image-processing-cpp
cpp implementation for algorithms in the book "数字图像处理与机器视觉-Visual C++与Matlab实现"

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

