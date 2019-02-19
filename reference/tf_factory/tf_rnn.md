# Recurrent Neural Network Model

1. Why standard network doesn't work

   - Inputs, outputs can be different lengths in different examples (the first and last layers)
   - Doesn't share features learned across different positions of text and hence too many parameters

2.  A basic RNN

   ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 67 - Recurrent Neural Network Model - deepl_ - https___www.coursera.org_learn_nlp.png)

   Although, now only info from earlier time is used for prediction

3. The computation procedure

   ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 68 - Backpropagation through time - deeplea_ - https___www.coursera.org_learn_nlp.png)

   The backward propagation (gradient procedure) is usually auto calculated inside framework ( like tf.GradientTape). Yet it basically goes backwards the computation route.

4.  Other usage and modified models

   - http://karpathy.github.io/2015/05/21/rnn-effectiveness/

   - Different type of problems

     ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 69 - Different types of RNNs - deeplearning_ - https___www.coursera.org_learn_nlp.png)

   - And their models

     ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 70 - Different types of RNNs - deeplearning_ - https___www.coursera.org_learn_nlp.png)

5. Vanishing gradients with RNNs

   - Traditional RNNs suffers vanishing gradients when the sentence goes longer. 

     ```
     '''
     Exapmle: 
     The cat, XXXXXXXX, was full.
     The cats, XXXXXXXX, were full.
     '''
     ```

   - Using GRU (gated recurrent unit) to 'remember' 

     ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 71 - Gated Recurrent Unit (GRU) - deeplearn_ - https___www.coursera.org_learn_nlp.png)

     C : cell, a vector with different dims to remember different information

     Gamma : gates to decide whether update of certain memory is needed (remember gate)

     Because most time, gamma is very small, the C could be remembered for a long time(not diminishing along with computation).

   - Or LSTM, with three gates, Gamma_u for update, and Gamma_f for forget, and Gamma_o for output

     ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 72 - Long Short Term Memory (LSTM) - deeple_ - https___www.coursera.org_learn_nlp.png)

     GRU could be regarded as a simplified model of LSTM

   6. Bidirectional RNN

      ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 73 - Bidirectional RNN - deeplearning.ai I _ - https___www.coursera.org_learn_nlp.png)

      bRNN + LSTM is commonly used in NLP

   7. Deep RNN

      ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 74 - Deep RNNs - deeplearning.ai I Coursera_ - https___www.coursera.org_learn_nlp.png)

      Because the RNN is already very complicated, generally there won't be too many deep-recurrent nodes


   8. Word embedding

      - Why embedding

        - Allowing the generalization across words

          ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 75 - Word Representation - deeplearning.ai _ - https___www.coursera.org_learn_nlp.png)

      -  How embedding

        - Featurized representation:

          ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 76 - Word Representation - deeplearning.ai _ - https___www.coursera.org_learn_nlp.png)

          (Using a t-SNE algorithm one could plot this N-D features into a 2-D plot)

          ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 77 - Word Representation - deeplearning.ai _ - https___www.coursera.org_learn_nlp.png)

      - How to use word embedding

        - Learn word embeddings from large text corpus. (1-100B words), or download pre-trained embedding online
        - Transfer embedding to new task with smaller training set (100k words), this also transfer the 10000-D sparse data into like 300-D dense data
        - Optional : continue to finetune the word embedding with new data

        - Using word embedding in analogy reasoning by comparing their similarity 

          ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 79 - Properties of word embeddings - deeple_ - https___www.coursera.org_learn_nlp.png)

      - Learn word embedding

        - The earlier way

          ![](D:\Data\PythonProjects\tf_fmri_meg\reference\tf_factory\FireShot Capture 80 - Learning word embeddings - deeplearnin_ - https___www.coursera.org_learn_nlp.png)

          - Note that the embedding matrix E is not learned by some PCA or similar ways, but carried out through a neural model (deep-neural leaning)
          - Instead of using the whole sentence, using a smaller and fixed context could work as well (Last x words, or the nearby 1 word)
