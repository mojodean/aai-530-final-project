# aai-530-final-project
# Project Title: TBD

This project is part of the AAI-530 course in the Applied Aritificial Intelligence Program at the University of San Diego (USD).

### Project Status: [Completed]

## Installation

Download the [Smart Home Dataset with weather information from Kaggle](https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information/data).

## Required libraries to be installed:

```python
import opendatasets as od
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
```

## Project Introduction & Objective

To design an IoT system that can detect anomalies and energy consumption in a smart home.

`530_IoT_project.ipynb` - Autoencoder
`Final_LSTM2.ipynb` - LSTM
`df_test_10.csv` - dataframe of the anomalies for the Tableau

### Project Team Members

- Payal Patel

- Vivian Perng

- Dean P. Simmer

### Methods Used

- Classification

- Autoencoder

- LSTM


### Technoogies

- Python

- Jupyter Notebook

- Keras/Tensorflow

### Project Description

This study explores smart home energy consumption patterns and the potential for anomaly detection and forecasting using machine learning techniques. Using a dataset of 509,910 observations from a smart meter, the analysis examines energy consumption across different household appliances and its correlation with weather-related variables. A deep learning autoencoder was employed for anomaly detection, and identified irregularities in energy use. Additionally, a Long-Short Term Memory (LSTM) model was used to predict future energy consumption, achieving 99.84% accuracy in its predictive trends across seven days.