# TFM
 
This repository contains all the necessary files to replicate the results exposed in my Master's degree Final Project: Bitcoin price prediction through Recurrent Neural Networks (RNN), in particular LSTM are used. This are a type of RNN that have some improvements to overcome the problem of the vanishing gradient.

### Technologies used

#### **Python**
All the prediction is done in Python using Keras, a special environment has to be build in Anaconda in order to used this without any problem as using it on the main environment will cause issues due to Numpy's lastest version is not compatible with Keras/Tensorflow. A new environments needs to be created where Numpy's version has to be 1.18.5 and Tensorflow was installed in its 2.3 version.

Only Keras, Tensorflow and Numpy should be installed in this environment to avoid problems, the rest of the dependencies will be downloaded automatically.

#### **R**
In this study the Diebold-Mariano test is used. The best package available to do this test is forecast on R so the data obtained through Python was then exported and used in R. Only this test was done through a small script called DMtest_corregido.R


### Important files
I have shared all of the scripts just to see the evolution of the model but the final model is acomplished through the folling script: **RedesNeuronalesRecurrentesMultiple3mesesRecusivoV4ModeloNuevoFuncion.py** were there are 2 parameters called *period* and *days* can be changed change the rolling window size and the prediction period. As this was done quickly to test the LSTM no function as been build.

A couple of variables, normally ended in 30, 60, 90, 120 shoud be renamed if the previous parameters are changed. This is clearly visible in the last lines were the values for the DM test are obtained.

### Future improvements
A function should be build in order to use this script in a more intuitive way. This wasn't done because of time constraints.

This is by no means production code, it's purpose was only to demonstrate that LSTM can be used as a tool to help with bitcoin price predictions.
