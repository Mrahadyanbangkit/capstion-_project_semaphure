# Machine Learning Codebase 

This folder stores the machine learning feature of our application. Listed below are the details about our folders.

### Semaphore Alphabet Classification 🚩

<p>In the semaphore alphabet classification folder, we store the dataset for different semaphore representations of the alphabet, the model building notebook, and the model.py file.</p>

**1. Dataset**

The dataset used is Semaphore Alphabet data from [Dataset](https://data.mendeley.com/datasets/tc5tnchrs2/) and also a dataset that we created ourselves by taking images of the semaphore representation for each letter.

**2. Machine Learning Model**

In Semaphore Alphabet Classification, we use a [Convolutional Neural Network (CNN)](https://www.tensorflow.org/tutorials/images/cnn) model to classify semaphore representations of the alphabet. Each letter is represented by a specific combination of flags or signals. The goal is to detect and classify the semaphore representation based on images.

We have modeled 26 classes corresponding to each letter of the alphabet. The dataset comprises images of semaphore signals representing each letter.The dataset comprises images of semaphore signals representing each letter. We split the dataset into training (80%) and testing (20%) sets to build and validate the model.

  **Model Architecture:**
  
   - Input Layer :  
     Image data representing semaphore representations.
   - Convolutional Layers:  
     Utilize convolutional layers to capture spatial features.
   - Pooling Layers :  
     Implement pooling layers for downsampling and feature selection.
   - Flatten Layer :  
     Flatten the output to feed into a dense layer.
   - Dense Layers :  
     Fully connected layers for classification.
   - Output Layer :  
     26 nodes representing each letter of the alphabe


The notebooks that we use can be accessed in [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1y6WgvWIS6GivMgv8crTgY6gZE9Kg_hmy?usp=sharing) 
  


