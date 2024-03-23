

https://github.com/alifatma622/Slash-AI-Internship/assets/101599059/5f0486ba-0f89-4859-890a-b34638c8b8ed

# Model Name (Product Image Classification with VGG16)
# Overview
This repository contains code for a deep learning model designed for image classification. The model is built using the VGG16 architecture as a base, with additional layers for classification on top. It is trained on a dataset containing images from three categories:
Accessories, Beauty, and Fashion. The primary goal of the model is to accurately classify images into these categories.
# Dataset
The dataset used for training and evaluation is located in the https://www.kaggle.com/datasets/fatmaelshihna/slash-dataset. It contains images of various accessories, beauty products,
and fashion items. The dataset is split into training and testing sets, with 80% of the images used for training and 20% for testing.
# Data Preparation
1. Define the path to the dataset directory: The path to the dataset directory is specified as /kaggle/input/slash-dataset/slash_dataset.
2. List the folders in the dataset directory: The categories in the dataset are listed as Accessories, Beauty, and Fashion.
3. Create lists for file paths and labels: File paths and corresponding labels are extracted from the dataset directory.
4. Split the dataset: The dataset is split into training and testing sets using a 80-20 split.
5. Print the number of samples in each category: The number of samples in each category for both the training and testing sets are printed.
6. Print the distribution of labels: The distribution of labels (categories) in both the training and testing sets is printed.
7. Load train and test images: Images are loaded as arrays and stored in separate lists for training and testing.

# Model Building
Base Model: The VGG16 model pretrained on ImageNet is used as the base model.
Additional Layers: Flatten, Dense, and Dropout layers are added on top of the base model for classification.
Model Compilation: The model is compiled with the Adam optimizer, categorical crossentropy loss function, and accuracy metric.

# Model Training
The model is trained using the fit method on the training data. Training is performed for 10 epochs, with validation data provided for monitoring performance.

# Model Evaluation
Evaluate on Train Dataset: The model is evaluated on the training dataset to assess its performance on data it has seen during training.
Evaluate on Test Dataset: The model is evaluated on the test dataset to assess its generalization performance on unseen data.

# Fine-Tuning
Fine-tuning is performed by unfreezing certain layers of the base model and retraining them with a lower learning rate. 
This is aimed at improving the model's performance on the this task.

# Additional Functionalities
1. Confusion Matrix: A confusion matrix is generated to visualize the performance of the model in terms of correct and incorrect classifications.
2. Loss and Accuracy Visualization: Loss and accuracy curves are plotted for both the training and testing sets to visualize the training process and identify any overfitting or underfitting 
  
