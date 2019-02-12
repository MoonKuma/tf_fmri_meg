1.  Compare similar models


| Evaluation | Meaning                                                      |
| ---------- | ------------------------------------------------------------ |
| Precision  | if classifier A has 95% precision, this means that when classifier A says something is a cat, there's a 95% chance it really is a cat |
| Recall     | if classifier A is 90% recall, this means that of all of the images in your dev sets that really are cats, classifier A accurately pulled out 90% of them |
| F1-score   | the harmonic mean of precision P and recall R, this is used as a single value to decide which model works better |

   ```python
   # compute F1
   def compute_F1(R, P):
   	return 2/(1/R + 1/P)
   ```

2. Train, develop and test dataset

   Through comparing how model behaves in train, dev and test set and help guide where to optimize next.

   For those conditions where the data in usage and data collected comes from different distribution( like hard to collect enough data from real apps so some other data from internet are used instead), one could also set two dev set as one from the train and one from the real-world condition

   ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 39 - Bias and Variance with mismatched data_ - https___www.coursera.org_learn_mac.png)



3. Compute the dimensions of conv result

   ```python
   # n*n image
   # f*f filter
   # padding p
   # stride s
   
   d_result = (np.floor((n+2*p-f)/s + 1), np.floor((n+2*p-f)/s + 1))
   
   ```


4. RGB, filters and it's result

![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 40 - Convolutions Over Volume - deeplearnin_ - https___www.coursera.org_learn_con.png)

- filters have the same number of channels/depth
- in the result after convolution, #channels(next layers) = #filters(used)
- Also. **the number of parameters (number of filters * weights and bias for each filter) is irrelevant with the size of image**



5. Pooling

   ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 41 - Pooling Layers - deeplearning.ai I Cou_ - https___www.coursera.org_learn_con.png)

- The largest number in certain area(if it is large) usually means a detection of certain feature during filter.
- There are no parameters to learn during pooling (a fix computation) 

![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 42 - CNN Example - deeplearning.ai I Course_ - https___www.coursera.org_learn_con.png)

- Actually most parameters come from the full connections layers



6. Why convolutions

   ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 43 - Why Convolutions_ - deeplearning.ai I _ - https___www.coursera.org_learn_con.png)

   - Significantly less parameters (irrelevant with the the size of pictures )

     - **Parameter sharing**:  a feature detector (i.e. a filter ) that is useful in one part of the image is probably useful in another part of the image

     -  **Sparsity of connections** : in each layers, each output value depends only on a small (localized) number of inputs



7. Classic convolution neural networks

   - LeNet

     ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 44 - Classic Networks - deeplearning.ai I C_ - https___www.coursera.org_learn_con.png)

   - AlexNet (the red parts were less often used in later work)

     ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 45 - Classic Networks - deeplearning.ai I C_ - https___www.coursera.org_learn_con.png)

   - VGG-16 (16 is for 16 layers)

     ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 46 - Classic Networks - deeplearning.ai I C_ - https___www.coursera.org_learn_con.png)



8. The inception neural network and its key parts

   - Residual block

     ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 47 - ResNets I Coursera_ - https___www.coursera.org_learn_con.png)

     Such residual blocks prevent the network from going worse instead of better when adding additional layers.

   - 1 * 1 convolution

     ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 48 - Inception Network Motivation - deeplea_ - https___www.coursera.org_learn_con.png)

     1 * 1 convolution is helpful in shrink down the representation like creating a bottleneck, and hence save the computation.

   - Inception module

     ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 50 - Inception Network I Coursera_ - https___www.coursera.org_learn_con.png)

   - Inception networks

     ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 51 - Inception Network I Coursera_ - https___www.coursera.org_learn_con.png)

     As stacking up inception modules.


   9.  Transfer learning

  ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 52 - Transfer Learning - deeplearning.ai I _ - https___www.coursera.org_learn_con.png)

   - Download others network and their weights, freeze part of it (make weights in those layers untrainable), and train the rest. 
   - The number of layers to freeze is dependent on how many data ( also how good the computer) one have

10.  On benchmarks and winning competitions

![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 53 - State of Computer Vision - deeplearnin_ - https___www.coursera.org_learn_con.png)



11. Object detection and localization

    - First build a detection model (like car or not)

    - Then use sliding window techniques ( by sliding the window at a small step each time and detect whether there are a car inside such window) to localize the object

    - Instead of doing the sliding by hand (which cost to much unnecessary computational cost), using a convolutional implementation

      - Turing FC layers into convolutional layers

        ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 55 - Convolutional Implementation of Slidin_ - https___www.coursera.org_learn_con.png)

      - Using that above as the detection network (14*14 here), and applying it to larger pics

        ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 56 - Convolutional Implementation of Slidin_ - https___www.coursera.org_learn_con.png)

        The key advantage of convolution implementation is that it saves many similar computation of the same area.

      - From localization to bounding box 

        - YOLO algorithm (instead of sliding, slicing the picture into different parts, so that one object won't appears in two localization)

          ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 57 - Bounding Box Predictions I Coursera_ - https___www.coursera.org_learn_con.png)

        - 