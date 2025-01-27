# Brain Tumor Detection using TensorFlow

This repository contains a TensorFlow-based model for detecting brain tumors in MRI images. The model achieves a high accuracy of 96% when trained and tested on a dedicated GPU.

## Introduction

Brain tumors pose a significant health challenge, and early detection is crucial for effective treatment. This project aims to develop an accurate and efficient automated system for brain tumor detection using deep learning. We leverage the power of TensorFlow and Convolutional Neural Networks (CNNs) to analyze MRI images and classify them as either containing a tumor or not.

## Abstract

This project utilizes a deep learning approach for brain tumor detection. A Convolutional Neural Network (CNN) is trained on a dataset of MRI images to classify images as either containing a tumor or being normal. The model is built using TensorFlow and optimized for GPU performance. The dataset used in this project is sourced from Kaggle. The dataset is split into training (70%), testing (10%), and validation (20%) sets to ensure robust model evaluation. The model achieves a test accuracy of 96% demonstrating its potential for assisting medical professionals in diagnosis.

## Dataset

The dataset used for training and evaluating this model was obtained from Kaggle. It consists of MRI images of the brain, labeled with the presence or absence of tumors. The dataset was split into the following proportions:

*   **Training Set:** 70% (3200 - 100 batch) of images
*   **Validation Set:** 20% (896 - 28 batch) of images
*   **Testing Set:** 10% (448 - 14 batch

To use dataset:
```python
import kagglehub
# Download latest version
path = kagglehub.dataset_download("preetviradiya/brian-tumor-dataset")
print("Path to dataset files:", path)
```

## Dependencies

To run this project, you will need the following software and Python packages:

### Software

*   **Python:** 3.12 or higher
*   **CUDA Toolkit and cuDNN:** CUDA 12, CuDNN 9 or higher
*   **TensorFlow with GPU support:** 2.18 or higher

### Python Packages

You can install the required packages using pip:

```bash
pip install tensorflow opencv-python matplotlib keras streamlit
```

## Running the Model and Testing

### Clone the repository

```bash
git clone https://github.com/shahriar-hd/BrainTumorDetection.git
```

Make sure you have your dataset in the correct directory as defined in your python script.

### Testing

To test the model on new images, you can use the provided testing script or integrate the model into your own application. Ensure the input images are preprocessed in the same way as the training data (e.g., resizing, normalization).

Example of testing code:

```python
img = cv2.imread('Cancer.jpg')
plt.imshow(img)
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show
yhat = model.predict(np.expand_dims(resize/255, 0))
if yhat > 0.5:
    print('Healthy')
    print(f'Accuracy: {yhat[0][0] * 100} %')
else:
    print('Cancer')
    print(f'Error percentage: {yhat[0][0] * 100} %')
```

### Output
The model outputs a probability score indicating the likelihood of a brain tumor being present in the input MRI image. A threshold can be applied to this score to classify the image as either positive (tumor detected) or negative (no tumor detected). </br>
During training the model will save the training history and the best model to the disk.

## License

This project is licensed under the [MIT License](https://github.com/shahriar-hd/BrainTumorDetection/blob/main/LICENSE).

## ontact
For any questions or inquiries, please feel free to contact me at [shahriar.hd@outlook.com](mailto:shahriar.hd@outlook.com)