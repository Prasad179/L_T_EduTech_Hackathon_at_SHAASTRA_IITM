# L&T EduTech Hackathon at SHAASTRA IITM

## Problem Statement 1:
Description: Concrete cracking is a major issue in Bridge Engineering. Detection of cracks facilitates the design, construction and maintenance of bridges effectively.

### Requirement Specification:
Develop a suitable Deep Learning framework which can detect the crack in the image from the dataset

### Judging Metrics:
Provide Precision, Recall and F1 score as Judging metrics

### Dataset Description:
There are 600 iamges which contains 300 images of cracked surface and 300 of non-cracked images. In the test and validation dataset there are 200 images in each dataset.

### Results and discussion:

From the literature, we have selected MobilenetV2 as our pretrained model, as it performs better than many other models, and it is light weight model. We have used fine tuning for our model. After fine tuning accuracy of model increases.

The following results are of test dataset. The accuracy of model is 98% and weighted F1 score is 98%.

| Class Name    |   Precision    |  Recall | F1 Score |
| ------------- | -------------- | --------|----------|
|  Non-cracked  |     0.97       |   1.0   |   0.99   |
|  Cracked      |     1.0        |   0.97  |   0.98   |



## Problem Statement 3:
### Description: 
Natural disasters and atmospheric anomalies demand remote monitoring and maintenance of naval objects especially big-size ships. For example, under poor weather conditions, prior knowledge about the ship model and type helps the automatic docking system process to be smooth. Thus, this set aims to classify the type of ships from an image data set of ships.

### Requirement Specification:
1. Design transfer learning-based CNN architecture to classify the data set
2. Identify an optimal training data size in percentage
### Judging Metrics: Kappa score
### Dataset Description:
There are 6252 images in train and 2680 images in test data. The categories of ships and their corresponding codes in the dataset are as follows -

There are 5 classes of ships to be detected which are as follows:

1. Cargo
2. Military
3. Carrier
4. Cruise
5. Tankers

To design transfer learning based CNN architecture, I have compared five pretrained CNN models. In those model I have not used fine tuning as it was resulting in low accuracy. To select the best among them, I have considered the weighted F1 score as a metric, as the problem is an imbalanced multi-class classification problem.

They are as follows:
1. MobilenetV2
2. Resnet152V2
3. Efficientnet
4. VGG19
5. Xception

Looking at the results, we can see that Resnet152V2 and VGG19 are out of competition as the best CNN architecture for our particular problem. MobilenetV2, Efficientnet, and Xception are giving nearly the same F1 score. MobilnetV2 and Xception are lightweight architectures. Between those two, I have selected Xception for further analysis.

### Results and discussion:

On the top of Xception model, I have added one Conv2D, one dropout layer to avoid overfitting, one GlobalAveragePooling2D and last layer is dense layer which gives probabilities of each class with the help of softmax activation function.

The developed model with **Xception** gives **accuracy** approximately around **92%-93%** and weighted **F1 score** around **0.92**. These results are after **20 epochs**. If we increase epochs, accuracy increases slightly.

To get optimum training dataset, I have splitted data as 90% training, 5% validation data and 5% test data. From the 90% data, I have splitted data from 10% to 80%.
The same Xception architecture is used for comparison. From the results one can observe that, there is no much difference in Kappa score for 70% training data and 80% training data. Hence we can conclude that optimum training data for training is approximately 70% of dataset.

**Kappa Score for 70% training data = 0.9465**

**Kappa Score for 80% training data = 0.9467**

Code for selection of pretrained model is in the PS_3_1.ipynb file. I have ran this code on Google Colab. To access data, we have to upload kaggle.json file in the content folder in Colab.

Code for optimum training dataset is in PS_3_2.ipynb file. In this code, the graph of Kappa score vs training data has mistake in X axis scale. It should be from 0.1 to 0.8. I have ran this code on Kaggle. I have directly added data from kaggle, we can add data in the notebook.

In the PS_3_2_modified.ipynb, I have corrected small mistake in graph and re-ran again. This is giving some interesting results.

**Kappa score for 60% training data = 0.9383**

**Kappa score for 70% training data = 0.9341**

**Kappa score for 80% training data = 0.9344**

As we can see from above results, Kappa score corresponding to 60% is slightly higher, but from previous run and this modified code, one can conclude that **70% training data** is optimum for model and accuracy can be increased by increasing epochs.
