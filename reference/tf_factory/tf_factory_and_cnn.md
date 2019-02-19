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

        - YOLO algorithm (instead of sliding, slicing the picture into different parts, so that one object (the center of it) won't appears in two boxes), You Only Look Once

          ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 57 - Bounding Box Predictions I Coursera_ - https___www.coursera.org_learn_con.png)

        - Anchor box - using more than one detection box incase of overlaping

        - Non-max suppression - areas which have the largest relative IOU

        ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 58 - YOLO Algorithm - deeplearning.ai I Cou_ - https___www.coursera.org_learn_con.png)

12. Face recognition

    - One-shot problem and learning similarity function

      - One-shot : learn from one example to recognize the person again

      - learning similarity function

        ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 59 - One Shot Learning - deeplearning.ai I _ - https___www.coursera.org_learn_con.png)

      - Using soft-max won work for new comers.

      - But such similarity function could be helpful for it is trained independent to the identity of people, and hence capable of accepting new ones

    - Train this through a Siamese Network

      ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 60 - Siamese Network - deeplearning.ai I Co_ - https___www.coursera.org_learn_con.png)

      Compute the same network and have it learnt that if the input pics are from different person the output also have larger difference

    - Triplet loss function

      - A - anchor
      - P - positive sample (different pics from same person)
      - N - negative sample (pics from different person)
      - alpha - margin value, this is to force the model to give different judgement of AP and AN

      ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 61 - Triplet Loss I Coursera_ - https___www.coursera.org_learn_con.png)

      [Caution] During training, one still need more than one images of the same person

      [Caution] According to the new loss functions, the model stop learning when criteria met 

      [Caution] Yet it's generally easier for the model to consider pics from different persons more different than from the same person, so some hard triplet examples are required for enhance the performance

    - Or using two pics and predict only whether they are from the same person

      ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 62 - Face Verification and Binary Classific_ - https___www.coursera.org_learn_con.png)

      For applying well-trained (fixed) model, pre-compute (store the encoding of the pictures instead of he raw pics) could save a lot of time.

13. Neural style transfer

    - Visualizing what a deep network is learning

      - 9 image patches that cause one of the hidden units(1 filer in layer 1) reach its maximum activation

        ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 63 - What are deep ConvNets learning_ - dee_ - https___www.coursera.org_learn_con.png)

      - As going deeper, the hidden units start to detect more complex textures

    - Neural style transfer cost function: the pics generated(G) should be both like the C pic in content and the S pic in style

      ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 64 - Cost Function - deeplearning.ai I Cour_ - https___www.coursera.org_learn_con.png)
      - Cost function of C,G - defined as the difference of the activation of certain layer

      - Cost function of S,G - defined as the correlation across channels in different layers

        - lambda is used to control whether the generated image will share the elementary style with the style image more, or those complex style  

        ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 66 - Style Cost Function - deeplearning.ai _ - https___www.coursera.org_learn_con.png)