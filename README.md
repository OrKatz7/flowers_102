# flowers_102
**Assignment No. 3**

**Part 1 – Aim: ANN from Scratch (40 Points)**

1) Read chapter 11: [“Implementing a Multi-layer Artificial Neural Network from Scratch” of the book “Machine Learning with PyTorch and Scikit-Learn](https://drive.google.com/file/d/1rzCDAkmLlOdqGciO7NnARlnahCeaqzFS/view?usp=sharing)” by Raschka et al. (2022)
1) Given the code of chapter 11 that can be found in:

<https://github.com/rasbt/machine-learning-book/blob/main/ch11/ch11.ipynb>

your main goal is to extend the code to address two hidden layers (instead of a single hidden layer). Extend the code by creating a local copy of the ch11.ipynb, perform the revisions, and submit the GitHub link to your revised code.

1) Apply the code of section 2 with the two layers for classifying handwritten digits MNIST dataset using the same full ANN architecture presented in the class (see “Solution 1: A plain deep NN”) and evaluate its prediction performance (macro AUC) using Train(70%)/Test(30%) validation procedure. 
1) Compare the predictive performance of section 3 with the original (single hidden layer) code and with the fully connected ANN implemented in Keras/TensorFlow/PyTorch (choose one).



Part 1:

Implementation - 2 hidden layer neural network

![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.001.png)

Forward function - 2 hidden layer neural network

![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.002.png)

Back propagation function - 2 hidden layer neural network

![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.003.png)

Updating the weights - 2 hidden layer neural network

Part 1 - Results:

![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.004.png)![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.005.png)

Network convergence - Left: model 1 with 1 hidden layer. Right: model 2 with 2 hidden layers.


|Model|<p>Model 1 </p><p>(1 hidden, Numpy)</p>|<p>Model 2</p><p>(2 hidden, Numpy)</p>|<p>Model 3</p><p>(1 hidden, Pythorch)</p>|
| :- | :- | :- | :- |
|Test accuracy|96.02%|95.9%|96.8%|
|Test AUC (Macro-average)|0.99586|0.99617|0.99620|
![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.006.png)

**Part 2 (60 points) - Aim: Practice the usage of CNN (Convolutional Neural Network).** 

In this part you will use at least two pretrained CNNs to identify the type of a flower that appears in an image. You will need to choose your pretrained models and use Transfer Learning to associate flower images into their corresponding categories.

For example – for the following image:

![C:\Users\Bar\Desktop\download (2).jpg](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.007.jpeg)






We would like the model to classify the image into its category (the dandelions category in this example). The model should be probabilistic and returns the probability of a flower belonging to each of the categories.

**General instructions:**

\1. The code should be written either in Python or R.

\2. We recommend that the code be implemented with one of the following deep-Learning packages: Keras/TensorFlow/PyTorch.

\3. Choose **at least** two pre-trained models (such as VGG) and adapt them to the current task.

\4. For basic training, use the following image database provided in: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ 

\5. Additional Images from other repositories can be added to improve accuracy.

\6. The dataset should be randomly divided into training (50%), validation (25%) for hyperparameter tuning, and test sets (25%). This random split should be repeated at least twice.

\7. Describe in detail the preprocessing process you performed to get the input from the raw images.

![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.008.png)

\8. Describe in detail the network you are using (including the specific layers).

Swin Transformer

Swin Transformer architecture splits an input RGB image into non-overlapping patches by a patch splitting module. Each patch is treated as a “token” and its feature is set as a concatenation of the raw pixel RGB values.

![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.009.png)

**Swin Transformer block** consists of a shifted window based MSA module, followed by a 2-layer MLP with GELU nonlinearity in between. A LayerNorm (LN) layer is applied before each MSA module and each MLP, and a residual connection is applied after each module. 

**Shifted window partitioning** in successive blocks The window-based self-attention module lacks connections across windows, which limits its modeling power. To introduce cross-window connections while maintaining the efficient computation of non-overlapping windows, we propose a shifted window partitioning approach which alternates between two partitioning configurations in consecutive Swin Transformer blocks

**W-MSA and SW-MSA** denote window based multi-head self-attention using regular and shifted window partitioning configurations, respectively.
# EfficientNet
EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. Unlike conventional practice that arbitrary scales these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients.

EfficientNet-B5 Architecture 

The first thing is any network is its stem after which all the experimenting with the architecture starts which is common in all the eight models and the final layers.

![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.010.png)

Module 1 — This is used as a starting point for the sub-blocks.

Module 2 — This is used as a starting point for the first sub-block of all the 7 main blocks except the 1st one.

Module 3 — This is connected as a skip connection to all the sub-blocks.

Module 4 — This is used for combining the skip connection in the first sub-blocks.

Module 5 — Each sub-block is connected to its previous sub-block in a skip connection and they are combined using this module.

\9. Provide an accuracy graph and the Cross-Entropy graph for train/validation/test as a function of the number of epochs for all models.

![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.011.png)

![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.012.png)

![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.013.png)

![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.014.png)

![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.015.png)

![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.016.png)

![](Aspose.Words.bd88cd54-6930-48b7-baf1-ca72d9b9e385.017.png)

**Minimum accuracy performance**

Accuracy in the test set of part 2 should be greater than 70% by at least one of the models.

Bonus: The team with the highest accuracy percentages (by at least one of the models) will receive a 7-point bonus, the 3 following teams will receive a 5-point bonus, and the next 5 teams will receive a 3-point bonus.

Submission should include:

One file (RAR or ZIP) should be submitted in Moodle (תרגיל 3). The file should contain:

1) Readme file with the links to the GitHub source codes of part 1 and part 2
1) A link to other Datasets (if you used them in addition to the database specified in section 3)
1) A PDF file for explaining your solution and results. 


