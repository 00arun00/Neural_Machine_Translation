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
- `Y`: a processed version of the machine readable dates in the training set, where each character is replaced by the index it is mapped to in `machine_vocab`. we should have `Y.shape = (m, Ty)`.
- `Xoh`: one-hot version of `X`, the "1" entry's index is mapped to the character thanks to `human_vocab`. `Xoh.shape = (m, Tx, len(human_vocab))`
- `Yoh`: one-hot version of `Y`, the "1" entry's index is mapped to the character thanks to `machine_vocab`. `Yoh.shape = (m, Tx, len(machine_vocab))`. Here, `len(machine_vocab) = 11` since there are 11 characters ('-' as well as 0-9).

## 2 - Neural machine translation with attention

While translating a book's paragraph from French to English, we would not read the whole paragraph, then close the book and translate. Even during the translation process, we would read/re-read and focus on the parts of the French paragraph corresponding to the parts of the English we are writing down.

The attention mechanism tells a Neural Machine Translation model where it should pay attention to at any step.


### 2.1 - Attention mechanism

Here is a figure that describes how the model works. The diagram on the left shows the attention model. The diagram on the right shows what one "Attention" step does to calculate the attention variables ![alpaha](http://latex.codecogs.com/gif.latex?%5Calpha%5E%7B%5Clangle%20t%2C%20t%27%20%5Crangle%7D), which are used to compute the context variable ![context](http://latex.codecogs.com/gif.latex?context%5E%7B%5Clangle%20t%20%5Crangle%7D) for each time-step in the output ![t](http://latex.codecogs.com/gif.latex?%24t%3D1%2C%20%5Cldots%2C%20T_y%24).

<table>
<td>
<img src="https://raw.githubusercontent.com/00arun00/Neural_Machine_Translation/master/images/attn_model.png" style="width:500;height:500px;"> <br>
</td>
<td>
<img src="https://raw.githubusercontent.com/00arun00/Neural_Machine_Translation/master/images/attn_mechanism.png" style="width:500;height:500px;"> <br>
</td>
</table>
<caption><center> **Figure 1**: Neural machine translation with attention</center></caption>



Here are some properties of the model:

- There are two separate LSTMs in this model (see diagram on the left). Because the one at the bottom of the picture is a Bi-directional LSTM and comes *before* the attention mechanism, we will call it *pre-attention* Bi-LSTM. The LSTM at the top of the diagram comes *after* the attention mechanism, so we will call it the *post-attention* LSTM. The pre-attention Bi-LSTM goes through ![Tx](http://latex.codecogs.com/gif.latex?T_x) time steps; the post-attention LSTM goes through ![Ty](http://latex.codecogs.com/gif.latex?T_y) time steps.

- We use ![bia](http://latex.codecogs.com/gif.latex?a%5E%7B%5Clangle%20t%20%5Crangle%7D%20%3D%20%5B%5Coverrightarrow%7Ba%7D%5E%7B%5Clangle%20t%20%5Crangle%7D%3B%20%5Coverleftarrow%7Ba%7D%5E%7B%5Clangle%20t%20%5Crangle%7D%5D) to represent the concatenation of the activations of both the forward-direction and backward-directions of the pre-attention Bi-LSTM.

- The diagram on the right uses a `RepeatVector` node to copy ![s_t-1](http://latex.codecogs.com/gif.latex?%24s%5E%7B%5Clangle%20t-1%20%5Crangle%7D%24)'s value ![T_x](http://latex.codecogs.com/gif.latex?%24T_x%24) times, and then `Concatenation` to concatenate ![s_t-1](http://latex.codecogs.com/gif.latex?%24s%5E%7B%5Clangle%20t-1%20%5Crangle%7D%24) and ![a_t](http://latex.codecogs.com/gif.latex?%24a%5E%7B%5Clangle%20t%20%5Crangle%7D%24) to compute ![e_t](http://latex.codecogs.com/gif.latex?%24e%5E%7B%5Clangle%20t%2C%20t%27%5Crangle%7D%24), which is then passed through a softmax to compute ![alpha_t_t](http://latex.codecogs.com/gif.latex?%24%5Calpha%5E%7B%5Clangle%20t%2C%20t%27%20%5Crangle%7D%24). We'll explain how to use `RepeatVector` and `Concatenation` in Keras below.

**1) `one_step_attention()`**: At step ![t](http://latex.codecogs.com/gif.latex?%24t%24), given all the hidden states of the Bi-LSTM ![a_ts](http://latex.codecogs.com/gif.latex?%24%5Ba%5E%7B%3C1%3E%7D%2Ca%5E%7B%3C2%3E%7D%2C%20...%2C%20a%5E%7B%3CT_x%3E%7D%5D%24) and the previous hidden state of the second LSTM (![s_t-1](http://latex.codecogs.com/gif.latex?%24s%5E%7B%3Ct-1%3E%7D%24)), `one_step_attention()` will compute the attention weights ![alphas](http://latex.codecogs.com/gif.latex?%24%5B%5Calpha%5E%7B%3Ct%2C1%3E%7D%2C%5Calpha%5E%7B%3Ct%2C2%3E%7D%2C%20...%2C%20%5Calpha%5E%7B%3Ct%2CT_x%3E%7D%5D%24) and output the context vector (see Figure  1 (right) for details):   
![eq1](http://latex.codecogs.com/gif.latex?%24%24context%5E%7B%3Ct%3E%7D%20%3D%20%5Csum_%7Bt%27%20%3D%200%7D%5E%7BT_x%7D%20%5Calpha%5E%7B%3Ct%2Ct%27%3E%7Da%5E%7B%3Ct%27%3E%7D%5Ctag%7B1%7D%24%24)

**2) `model()`**: Implements the entire model. It first runs the input through a Bi-LSTM to get back ![ats](http://latex.codecogs.com/gif.latex?%24%5Ba%5E%7B%3C1%3E%7D%2Ca%5E%7B%3C2%3E%7D%2C%20...%2C%20a%5E%7B%3CT_x%3E%7D%5D%24). Then, it calls `one_step_attention()` ![ty](http://latex.codecogs.com/gif.latex?%24T_y%24) times (`for` loop). At each iteration of this loop, it gives the computed context vector ![ct](http://latex.codecogs.com/gif.latex?%24c%5E%7B%3Ct%3E%7D%24) to the second LSTM, and runs the output of the LSTM through a dense layer with softmax activation to generate a prediction ![y_hat_t](http://latex.codecogs.com/gif.latex?%5Chat%7By%7D%5E%7B%3Ct%3E%7D).

we train the model using `categorical_crossentropy` loss, a custom [Adam](https://keras.io/optimizers/#adam) [optimizer](https://keras.io/optimizers/#usage-of-optimizers) (`learning rate = 0.005`, ![alphas](http://latex.codecogs.com/gif.latex?%24%5Cbeta_1%20%3D%200.9%24), ![alphas](http://latex.codecogs.com/gif.latex?%24%5Cbeta_2%20%3D%200.9%24), `decay = 0.01`)  and `['accuracy']` metrics:

While training we can see the loss as well as the accuracy on each of the 10 positions of the output. The table below gives us an example of what the accuracies could be if the batch had 2 examples:

<img src="https://raw.githubusercontent.com/00arun00/Neural_Machine_Translation/master/images/table.png" style="width:700;height:200px;"> <br>



## Results
Sample translations
```
source: 3 May 1979
output: 1979-05-03
source: 5 April 09
output: 2009-05-05
source: 21th of August 2016
output: 2016-08-21
source: Tue 10 Jul 2007
output: 2007-07-10
source: Saturday May 9 2018
output: 2018-05-09
source: March 3 2001
output: 2001-03-03
source: March 3rd 2001
output: 2001-03-03
source: 1 March 2001
output: 2001-03-01
```
Accuracy
```
 'dense_3_acc_1': [1.0],
 'dense_3_acc_10': [0.99970000028610229],
 'dense_3_acc_2': [1.0],
 'dense_3_acc_3': [1.0],
 'dense_3_acc_4': [1.0],
 'dense_3_acc_5': [1.0],
 'dense_3_acc_6': [0.99890000104904175],
 'dense_3_acc_7': [0.99980000019073489],
 'dense_3_acc_8': [1.0],
 'dense_3_acc_9': [0.99620000362396244],
```
Losses
```
 'dense_3_loss_1': [0.0014263137261150405],
 'dense_3_loss_10': [0.002527753144968301],
 'dense_3_loss_2': [0.00033948474334465572],
 'dense_3_loss_3': [0.00089316421450348572],
 'dense_3_loss_4': [0.0026649332512170077],
 'dense_3_loss_5': [1.8729131916188635e-05],
 'dense_3_loss_6': [0.005103303827054333],
 'dense_3_loss_7': [0.0068005907908082006],
 'dense_3_loss_8': [3.4973986021213933e-06],
 'dense_3_loss_9': [0.019907007873989642],
 'loss': [0.039684777855873106]
```

## Visualizing Attention

<img src="https://raw.githubusercontent.com/00arun00/Neural_Machine_Translation/master/images/attention_map.png" style="width:500;height:436px;"> <br>
