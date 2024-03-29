# TFM
 
This repository contains all the necessary files to replicate the results exposed in my Master's degree Final Project: Bitcoin price prediction through Recurrent Neural Networks (RNN), in particular LSTM are used. This are a type of RNN that have some improvements to overcome the problem of the vanishing gradient.

### Technologies used

#### **Python**
All the prediction is done in Python using Keras, a special environment has to be build in Anaconda in order to used this without any problem as using it on the main environment will cause issues due to Numpy's lastest version is not compatible with Keras/Tensorflow. A new environments needs to be created where Numpy's version has to be 1.18.5 and Tensorflow was installed in its 2.3 version.

Only Keras, Tensorflow and Numpy should be installed in this environment to avoid problems, the rest of the dependencies will be downloaded automatically.

#### **R**
In this study the Diebold-Mariano test is used. The best package available to do this test is *forecast* on R so the data obtained through Python was then exported and used in R. Only this test was done through a small script called DMtest_corregido.R


### Important files
I have shared all of the scripts just to see the evolution of the model but the final model is acomplished through the folling script: **LSTM_90_days_rolling_window.py** were there are 2 parameters called *period* and *days* can be changed change the rolling window size (*period*) and the prediction period (*days*). For this study the parameter *days* will stay always the same as the prediction period will always be 90 days.

The rest of the files located in the main folder are a copy of this script changing the values of the *period* parameter for 30, 60 and 120.

For the 120 days model another .csv is used as the first data file (**BTC_val.csv**) had not enough data point at the beggining, so 30 more days had been added in the **BTC_val120.csv** to be able to use rolling window properly, this file starts at 30th of January and ends at 30th of November of 2021 instead of the previous file that ranges from 1st of March until 30th of November of 2021.

As this was done quickly to test the LSTM capabilities no function as been build.

A couple of variables, normally ended in 30, 60, 90, 120 should be renamed if the previous parameters are changed. This is clearly visible in the last lines were the values for the DM test are obtained.

### Future improvements
A function should be build in order to use this script in a more intuitive way. This wasn't done because of time constraints.

This is by no means production code, it's purpose was only to demonstrate that LSTM can be used as a tool to help with bitcoin price predictions.
