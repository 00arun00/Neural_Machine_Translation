# Neural Machine Translation

This project focuses on building a Neural Machine Translation (NMT) model to translate human readable dates ("25th of June, 2009") into machine readable dates ("2009-06-25"), using  attention model.

# Steps

## 1 - Translating human readable dates into machine readable dates

Attention models can be used to translate from one language to another, such as translating from English to Hindi. However, language translation requires massive datasets and usually takes days of training on GPUs. This project focuses on getting the concepts required instead and hence we will use a simpler "date translation" task. 

The network will input a date written in a variety of possible formats (*e.g. "the 29th of August 1958", "03/30/1968", "24 JUNE 1987"*) and translate them into standardized, machine readable dates (*e.g. "1958-08-29", "1968-03-30", "1987-06-24"*). We will have the network learn to output dates in the common machine-readable format YYYY-MM-DD. 

### 1.1 - Dataset

The model here is trained on a dataset of 10000 human readable dates and their equivalent, standardized, machine readable dates. Data is preprocessed. Next  a map of the raw text data into the index values is created. Here we use Tx=30 (which we assume is the maximum length of the human readable date; if we get a longer input, we would have to truncate it) and Ty=10 (since "YYYY-MM-DD" is 10 characters long). 

After Preprocessing our dataset is of the following format:

- `X`: a processed version of the human readable dates in the training set, where each character is replaced by an index mapped to the character via `human_vocab`. Each date is further padded to ![Tx](http://latex.codecogs.com/gif.latex?T_x) values with a special character (< pad >). `X.shape = (m, Tx)`
- `Y`: a processed version of the machine readable dates in the training set, where each character is replaced by the index it is mapped to in `machine_vocab`. You should have `Y.shape = (m, Ty)`. 
- `Xoh`: one-hot version of `X`, the "1" entry's index is mapped to the character thanks to `human_vocab`. `Xoh.shape = (m, Tx, len(human_vocab))`
- `Yoh`: one-hot version of `Y`, the "1" entry's index is mapped to the character thanks to `machine_vocab`. `Yoh.shape = (m, Tx, len(machine_vocab))`. Here, `len(machine_vocab) = 11` since there are 11 characters ('-' as well as 0-9). 

## 2 - Neural machine translation with attention

While translating a book's paragraph from French to English, we would not read the whole paragraph, then close the book and translate. Even during the translation process, we would read/re-read and focus on the parts of the French paragraph corresponding to the parts of the English we are writing down. 

The attention mechanism tells a Neural Machine Translation model where it should pay attention to at any step. 


### 2.1 - Attention mechanism

We will implement the attention mechanism presented in the lecture videos. Here is a figure to remind you how the model works. The diagram on the left shows the attention model. The diagram on the right shows what one "Attention" step does to calculate the attention variables ![alpaha](http://latex.codecogs.com/gif.latex?%5Calpha%5E%7B%5Clangle%20t%2C%20t%27%20%5Crangle%7D), which are used to compute the context variable ![context](http://latex.codecogs.com/gif.latex?context%5E%7B%5Clangle%20t%20%5Crangle%7D) for each time-step in the output (![t](http://latex.codecogs.com/gif.latex?%24t%3D1%2C%20%5Cldots%2C%20T_y%24)). 

<table>
<td> 
<img src="https://raw.githubusercontent.com/00arun00/Neural_Machine_Translation/master/images/attn_model.png" style="width:500;height:500px;"> <br>
</td> 
<td> 
<img src="https://raw.githubusercontent.com/00arun00/Neural_Machine_Translation/master/images/attn_mechanism.png" style="width:500;height:500px;"> <br>
</td> 
</table>
<caption><center> **Figure 1**: Neural machine translation with attention</center></caption>



Here are some properties of the model that you may notice: 

- There are two separate LSTMs in this model (see diagram on the left). Because the one at the bottom of the picture is a Bi-directional LSTM and comes *before* the attention mechanism, we will call it *pre-attention* Bi-LSTM. The LSTM at the top of the diagram comes *after* the attention mechanism, so we will call it the *post-attention* LSTM. The pre-attention Bi-LSTM goes through ![Tx](http://latex.codecogs.com/gif.latex?T_x) time steps; the post-attention LSTM goes through ![Ty](http://latex.codecogs.com/gif.latex?T_y) time steps. 

- The post-attention LSTM passes ![sc](http://latex.codecogs.com/gif.latex?s%5E%7B%5Clangle%20t%20%5Crangle%7D%2C%20c%5E%7B%5Clangle%20t%20%5Crangle%7D) from one time step to the next. In the lecture videos, we were using only a basic RNN for the post-activation sequence model, so the state captured by the RNN output activations ![s](http://latex.codecogs.com/gif.latex?s%5E%7B%5Clangle%20t%5Crangle%7D). But since we are using an LSTM here, the LSTM has both the output activation ![s](http://latex.codecogs.com/gif.latex?s%5E%7B%5Clangle%20t%5Crangle%7D) and the hidden cell state ![ct](http://latex.codecogs.com/gif.latex?%24c%5E%7B%5Clangle%20t%5Crangle%7D%24). However, unlike previous text generation examples (such as Dinosaurus in week 1), in this model the post-activation LSTM at time ![t](http://latex.codecogs.com/gif.latex?%24t%24) does will not take the specific generated ![aba](http://latex.codecogs.com/gif.latex?%24y%5E%7B%5Clangle%20t-1%20%5Crangle%7D%24) as input; it only takes ![s_t](http://latex.codecogs.com/gif.latex?%24s%5E%7B%5Clangle%20t%5Crangle%7D%24) and ![c_t](http://latex.codecogs.com/gif.latex?%24c%5E%7B%5Clangle%20t%5Crangle%7D%24) as input. We have designed the model this way, because (unlike language generation where adjacent characters are highly correlated) there isn't as strong a dependency between the previous character and the next character in a YYYY-MM-DD date. 

- We use ![bia](http://latex.codecogs.com/gif.latex?a%5E%7B%5Clangle%20t%20%5Crangle%7D%20%3D%20%5B%5Coverrightarrow%7Ba%7D%5E%7B%5Clangle%20t%20%5Crangle%7D%3B%20%5Coverleftarrow%7Ba%7D%5E%7B%5Clangle%20t%20%5Crangle%7D%5D) to represent the concatenation of the activations of both the forward-direction and backward-directions of the pre-attention Bi-LSTM. 

- The diagram on the right uses a `RepeatVector` node to copy ![s_t-1](http://latex.codecogs.com/gif.latex?%24s%5E%7B%5Clangle%20t-1%20%5Crangle%7D%24)'s value ![T_x](http://latex.codecogs.com/gif.latex?%24T_x%24) times, and then `Concatenation` to concatenate ![s_t-1](http://latex.codecogs.com/gif.latex?%24s%5E%7B%5Clangle%20t-1%20%5Crangle%7D%24) and ![a_t](http://latex.codecogs.com/gif.latex?%24a%5E%7B%5Clangle%20t%20%5Crangle%7D%24) to compute ![e_t](http://latex.codecogs.com/gif.latex?%24e%5E%7B%5Clangle%20t%2C%20t%27%5Crangle%7D%24), which is then passed through a softmax to compute ![alpha_t_t](http://latex.codecogs.com/gif.latex?%24%5Calpha%5E%7B%5Clangle%20t%2C%20t%27%20%5Crangle%7D%24). We'll explain how to use `RepeatVector` and `Concatenation` in Keras below. 

Lets implement this model. You will start by implementing two functions: `one_step_attention()` and `model()`.

**1) `one_step_attention()`**: At step <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936155500000004pt height=20.222069999999988pt/>, given all the hidden states of the Bi-LSTM (![a_ts](http://latex.codecogs.com/gif.latex?%24%5Ba%5E%7B%3C1%3E%7D%2Ca%5E%7B%3C2%3E%7D%2C%20...%2C%20a%5E%7B%3CT_x%3E%7D%5D%24)) and the previous hidden state of the second LSTM (![s_t-1](http://latex.codecogs.com/gif.latex?%24s%5E%7B%3Ct-1%3E%7D%24)), `one_step_attention()` will compute the attention weights (![alphas](http://latex.codecogs.com/gif.latex?%24%5B%5Calpha%5E%7B%3Ct%2C1%3E%7D%2C%5Calpha%5E%7B%3Ct%2C2%3E%7D%2C%20...%2C%20%5Calpha%5E%7B%3Ct%2CT_x%3E%7D%5D%24)) and output the context vector (see Figure  1 (right) for details):
![eq1](http://latex.codecogs.com/gif.latex?%24%24context%5E%7B%3Ct%3E%7D%20%3D%20%5Csum_%7Bt%27%20%3D%200%7D%5E%7BT_x%7D%20%5Calpha%5E%7B%3Ct%2Ct%27%3E%7Da%5E%7B%3Ct%27%3E%7D%5Ctag%7B1%7D%24%24)

Note that we are denoting the attention in this notebook ![context](http://latex.codecogs.com/gif.latex?%24context%5E%7B%5Clangle%20t%20%5Crangle%7D%24). In the lecture videos, the context was denoted ![context](http://latex.codecogs.com/gif.latex?%24c%5E%7B%5Clangle%20t%20%5Crangle%7D%24), but here we are calling it ![context](http://latex.codecogs.com/gif.latex?%24context%5E%7B%5Clangle%20t%20%5Crangle%7D%24) to avoid confusion with the (post-attention) LSTM's internal memory cell variable, which is sometimes also denoted ![context](http://latex.codecogs.com/gif.latex?%24c%5E%7B%5Clangle%20t%20%5Crangle%7D%24). 

**2) `model()`**: Implements the entire model. It first runs the input through a Bi-LSTM to get back <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/045401048b3f4df2a84e54a6e44fc157.svg?invert_in_darkmode" align=middle width=163.12345499999998pt height=27.656969999999987pt/>. Then, it calls `one_step_attention()` <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/48fa77711600a61ef8d573289524e0f9.svg?invert_in_darkmode" align=middle width=16.685790000000004pt height=22.46574pt/> times (`for` loop). At each iteration of this loop, it gives the computed context vector <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/25365737389fbe8c187fbc8c988d9569.svg?invert_in_darkmode" align=middle width=32.62776pt height=26.086169999999992pt/> to the second LSTM, and runs the output of the LSTM through a dense layer with softmax activation to generate a prediction <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/b200dddfa9ac2f110ab9ca4e04caddd6.svg?invert_in_darkmode" align=middle width=34.163085pt height=26.086169999999992pt/>. 



**Exercise**: Implement `one_step_attention()`. The function `model()` will call the layers in `one_step_attention()` <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/48fa77711600a61ef8d573289524e0f9.svg?invert_in_darkmode" align=middle width=16.685790000000004pt height=22.46574pt/> using a for-loop, and it is important that all <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/48fa77711600a61ef8d573289524e0f9.svg?invert_in_darkmode" align=middle width=16.685790000000004pt height=22.46574pt/> copies have the same weights. I.e., it should not re-initiaiize the weights every time. In other words, all <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/48fa77711600a61ef8d573289524e0f9.svg?invert_in_darkmode" align=middle width=16.685790000000004pt height=22.46574pt/> steps should have shared weights. Here's how you can implement layers with shareable weights in Keras:
1. Define the layer objects (as global variables for examples).
2. Call these objects when propagating the input.

We have defined the layers you need as global variables. Please run the following cells to create them. Please check the Keras documentation to make sure you understand what these layers are: [RepeatVector()](https://keras.io/layers/core/#repeatvector), [Concatenate()](https://keras.io/layers/merge/#concatenate), [Dense()](https://keras.io/layers/core/#dense), [Activation()](https://keras.io/layers/core/#activation), [Dot()](https://keras.io/layers/merge/#dot).

Now you can use these layers to implement `one_step_attention()`. In order to propagate a Keras tensor object X through one of these layers, use `layer(X)` (or `layer([X,Y])` if it requires multiple inputs.), e.g. `densor(X)` will propagate X through the `Dense(1)` layer defined above.

You will be able to check the expected output of `one_step_attention()` after you've coded the `model()` function.

**Exercise**: Implement `model()` as explained in figure 2 and the text above. Again, we have defined global layers that will share weights to be used in `model()`.



Now you can use these layers <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/48fa77711600a61ef8d573289524e0f9.svg?invert_in_darkmode" align=middle width=16.685790000000004pt height=22.46574pt/> times in a `for` loop to generate the outputs, and their parameters will not be reinitialized. You will have to carry out the following steps: 

1. Propagate the input into a [Bidirectional](https://keras.io/layers/wrappers/#bidirectional) [LSTM](https://keras.io/layers/recurrent/#lstm)
2. Iterate for <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/68e160bac19c44808554044bddb171d2.svg?invert_in_darkmode" align=middle width=118.4205pt height=22.46574pt/>: 
    1. Call `one_step_attention()` on $[\alpha^{<t,1>},\alpha^{<t,2>}, ..., \alpha^{<t,T_x>}]$ and $s^{<t-1>}$ to get the context vector $context^{<t>}$.
    2. Give $context^{<t>}$ to the post-attention LSTM cell. Remember pass in the previous hidden-state $s^{\langle t-1\rangle}$ and cell-states $c^{\langle t-1\rangle}$ of this LSTM using `initial_state= [previous hidden state, previous cell state]`. Get back the new hidden state $s^{<t>}$ and the new cell state $c^{<t>}$.
    3. Apply a softmax layer to $s^{<t>}$, get the output. 
    4. Save the output by adding it to the list of outputs.
3. Create your Keras model instance, it should have three inputs ("inputs", <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/f8fd324ac4f409cff12924fd28ba24ca.svg?invert_in_darkmode" align=middle width=34.806090000000005pt height=26.76201000000001pt/> and <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/0b6a3548736097aa71e1abda128b5ac7.svg?invert_in_darkmode" align=middle width=34.214400000000005pt height=26.76201000000001pt/>) and output the list of "outputs".

As usual, after creating your model in Keras, you need to compile it and define what loss, optimizer and metrics your are want to use. Compile your model using `categorical_crossentropy` loss, a custom [Adam](https://keras.io/optimizers/#adam) [optimizer](https://keras.io/optimizers/#usage-of-optimizers) (`learning rate = 0.005`, <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/d3cdf08fd9338b44d5fc287a1ce6bcc8.svg?invert_in_darkmode" align=middle width=59.5947pt height=22.831379999999992pt/>, <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/d28315fe44f65564b4025eed2d551807.svg?invert_in_darkmode" align=middle width=76.033155pt height=22.831379999999992pt/>, `decay = 0.01`)  and `['accuracy']` metrics:

The last step is to define all your inputs and outputs to fit the model:

- You already have X of shape <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/fa242122b24793171b7bc3491822b9c1.svg?invert_in_darkmode" align=middle width=153.7767pt height=24.65759999999998pt/> containing the training examples.
- You need to create `s0` and `c0` to initialize your `post_activation_LSTM_cell` with 0s.
- Given the `model()` you coded, you need the "outputs" to be a list of 11 elements of shape (m, T_y). So that: `outputs[i][0], ..., outputs[i][Ty]` represent the true labels (characters) corresponding to the <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/3def24cf259215eefdd43e76525fb473.svg?invert_in_darkmode" align=middle width=18.325065000000002pt height=27.91271999999999pt/> training example (`X[i]`). More generally, `outputs[i][j]` is the true label of the <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/95291b39ba5d9dba052b40bf07b12cd2.svg?invert_in_darkmode" align=middle width=20.372220000000002pt height=27.91271999999999pt/> character in the <img src="https://rawgit.com/00arun00/Neural_Machine_Translation/None/svgs/3def24cf259215eefdd43e76525fb473.svg?invert_in_darkmode" align=middle width=18.325065000000002pt height=27.91271999999999pt/> training example.

While training you can see the loss as well as the accuracy on each of the 10 positions of the output. The table below gives you an example of what the accuracies could be if the batch had 2 examples: 

<img src="https://raw.githubusercontent.com/00arun00/Neural_Machine_Translation/master/images/table.png" style="width:700;height:200px;"> <br>
<caption><center>Thus, `dense_2_acc_8: 0.89` means that you are predicting the 7th character of the output correctly 89% of the time in the current batch of data. </center></caption>

We have run this model for longer, and saved the weights. Run the next cell to load our weights. (By training a model for several minutes, you should be able to obtain a model of similar accuracy, but loading our model will save you time.) 

