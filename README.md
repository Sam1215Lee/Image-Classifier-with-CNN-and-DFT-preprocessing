# Image-Classifier-with-CNN-and-DFT-preprocessing

## Project Description
The idea behind this project stems from the potential of DFT to reveal additional patterns or features not easily accessible in the spatial domain of images. By transforming images into their frequency domain, it is hypothesized that the model can gain a more comprehensive understanding of the images, possibly leading to improved classification accuracy.

## Installation
require libray
- opencv-python
- matplotlib
- tensorflow
- Pillow
- numpy
- scikit-learn
- seaborn
- pandas

## Dataset
The dataset should consist of grayscale images divided into four categories: cloudy, rain, shine, and sunrise. Place the training and test datasets in separate directories for each category.  
| label | cloudy | rain | sunrise |
|:-----:|:-----:|:-----:|:-----:|
|0|1|2|3|

## Model
![ALT](https://github.com/Potassium-chromate/CNN-for-recognizer-weather/blob/main/Picture/Model%20structure.png)
## Running the Code
To run the code, provide the appropriate paths for the training and test datasets. Also, specify the desired target size for resizing the images. The model is trained using the training dataset and evaluated on the test dataset. The training and validation accuracy and loss are plotted. The confusion matrix is also displayed for both the training and test sets.

The code also includes data augmentation (rotation and flipping) in the load function, but it's currently not being used. You can utilize it by passing 'yes' as the argument while loading the data.

## Results
### RGB
|       |train_acc|train_loss|test_acc|test_loss|
|:-----:|:-------:|:--------:|:------:|:-------:|
|CNN    |0.937    | 0.14     | 0.9067 |  0.8187 |  
|DFT + CNN|0.9494    | 0.14     | 0.92 |  0.698 |  
#### CNN confusion matrix
![ALT](https://github.com/Potassium-chromate/Image-Classifier-with-CNN-and-DFT-preprocessing/blob/main/picture/RGB/CNN_RGB%20confusion_test.png)
#### DFT+CNN confusion matrix
![ALT](https://github.com/Potassium-chromate/Image-Classifier-with-CNN-and-DFT-preprocessing/blob/main/picture/RGB/DFT_RGB%20confusion_test.png)
### Gray
|       |train_acc|train_loss|test_acc|test_loss|
|:-----:|:-------:|:--------:|:------:|:-------:|
|CNN    |0.9656   | 0.1078   | 0.7467 |  1.2034 |  
|DFT + CNN|0.9055 | 0.2641   | 0.7733 |  0.6454 |  
#### CNN confusion matrix
![ALT](https://github.com/Potassium-chromate/Image-Classifier-with-CNN-and-DFT-preprocessing/blob/main/picture/GRAY/CNN%20confusion_test.png)
#### DFT+CNN confusion matrix
![ALT](https://github.com/Potassium-chromate/Image-Classifier-with-CNN-and-DFT-preprocessing/blob/main/picture/GRAY/DFT%20confusion_test.png)

## Discussion
The DFT (Discrete Fourier Transform) is a tool used to analyze the frequency components of digital signals. In image processing, the 2D version of DFT is used which transforms the image from its spatial domain to its frequency domain.

Each pixel in an image represents a particular intensity to be displayed. But if we perform DFT on the image, each pixel will now represent a particular frequency contained in the spatial domain image. Low frequencies in the Fourier domain image correspond to slow changes in the spatial domain image, while high frequencies correspond to fast changes.

When applying the DFT to an image, you're essentially revealing information that wasn't easily accessible in the time/space domain. Patterns, details, and structures in the frequency domain might help the model to learn and make decisions better, which is why applying DFT to the image might lead to better model performance.







