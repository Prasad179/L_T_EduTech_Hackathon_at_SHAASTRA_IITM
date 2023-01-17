# L&T EduTech Hackathon at SHAASTRA IITM
### Problem Statement 3:
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

To design transfer learning based CNN architecture, I have compared five pretrained CNN models. To select the best among them, I have considered the weighted F1 score as a metric, as the problem is an imbalanced multi-class classification problem.

They are as follows:
1. MobilenetV2
2. Resnet152V2
3. Efficientnet
4. VGG19
5. Xception

