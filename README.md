# Anomaly-Detection
Anomaly Detection on Time-Series Sensor Measurement Data by using Unsupervised Machine Leatning and Deep Learning Methods

----

## DATA
JSON Data has 3 features including a timestamp and 2 sensor measurement.  The goal is to detect anomalies of sensor mesurement seperately from that time-series data which means making an time-series unvariant anomaly detection.
 
 
![alttext](Figures/Feature1-Ts.png) 
![alttext](Figures/Feature2-Ts.png) 

## METHODS AND RESULTS

### Local Outlier Factor
Among kenchi outlier detection, OneClassSVM, IsolationForest and LocalOutlierFactor models, LocalOutlierFactor gave the best results

![alttext](Figures/LOFonF1.png) 
![alttext](Figures/LOFonF2.png) 


### LUMINOL
Luminol is a light weight python library for time series data analysis. The two major functionalities it supports are anomaly detection and correlation.

Luminol Source: https://github.com/linkedin/luminol

![alttext](Figures/LUMonF1.png) 
![alttext](Figures/LUMonF2.png) 

## Robust Random Cut Forest Algorithm
The Robust Random Cut Forest (RRCF) algorithm is an ensemble method for detecting outliers in streaming data.

RRCF Source: https://github.com/kLabUM/rrcf 

![alttext](Figures/RRCFonF1.png) 
![alttext](Figures/RRCFonF2.png) 

### LSTM AutoEncoder
An LSTM Autoencoder is an implementation of an autoencoder for sequence data using an Encoder-Decoder LSTM architecture. LSTM Autoencoder can detect anomalies by reconstructing given data from a time range and comparing the original one.


![alttext](Figures/LSTM_AE_Training.png) 

![alttext](Figures/AEonF1.png) 
![alttext](Figures/AEonF2.png) 

