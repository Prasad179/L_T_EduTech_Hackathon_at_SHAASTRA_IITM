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

To design transfer learning based CNN architecture, I have compared five pretrained CNN models. In those model I have not used fine tuning as it was resulting in low accuracy. To select the best among them, I have considered the weighted F1 score as a metric, as the problem is an imbalanced multi-class classification problem.

They are as follows:
1. MobilenetV2
2. Resnet152V2
3. Efficientnet
4. VGG19
5. Xception

Looking at the results, we can see that, Resnet152V2 and VGG19 are out of competition as best CNN architecture for our particular problem. MobilenetV2, Efficientnet and Xception are giving nearly same F1 score. MobilnetV2 and Xception are lightweight architechture. Between those two, I have selected Xception for further analysis. 

### Results and discussion:

On the top of Xception model, I have added one Conv2D, one dropout layer to avoid overfitting, one GlobalAveragePooling2D and last layer is dense layer which gives probabilities of each class with the help of softmax activation function.

The developed model with Xception gives accuracy approximately around 92%-93% and weighted F1 score around 0.92. These results are after 20 epochs. If we increase epochs, accuracy increases slightly.
