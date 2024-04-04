# Tweet Sentiment Analysis

This is a simple Flask web application that analyzes whether a tweet has a positive or negative impact using a pre-trained sentiment analysis model.

## Overview

The application takes a tweet as input from the user and then predicts whether the sentiment of the tweet is positive or negative based on the text content. The sentiment analysis model is trained using a Naive Bayes classifier on a dataset of labeled tweets.

## Usage

To use this project:

1. Clone this repository to your local machine.
 
2. Install the necessary dependencies (Python, required libraries, etc.).
 
3. Explore the Jupyter notebooks in the notebooks/ directory to understand the project workflow.

4. Execute the source code model.py in the project/ directory to train the model.

5. Make sur to open all folder project/ content and execute the source code app.py to go on the web site of model deployment.

6. Start the Flask application 

7. Open your web browser and go to `http://localhost:5000` to access the application.

8. Enter a tweet in the input field provided and click the "Analyse" button.

9. The application will display whether the tweet has a positive or negative impact.

Feel free to open issues for any questions or suggestions!

Enjoy

## Model Training

The sentiment analysis model used in this application is trained using a Multinomial Naive Bayes classifier. The training data consists of a dataset of labeled tweets, which is preprocessed to clean the text data before training the model.

## Directory Structure

- **project/**:Contains the final source code for model implementation.
- **data/**: Contains the dataset used for training and evaluation.
- **notebooks/**: Includes Jupyter notebooks used for data exploration, model development, and evaluation.
- **README.md**: This file, providing an overview of the project, its objectives, methodology, and directory structure.

## Credits

- The sentiment analysis model is trained using the https://www.kaggle.com/datasets/dineshpiyasamara/sentiment-analysis-dataset.

## Project Made By

[Abdessamad Makan](https://github.com/abdessamad-makan)
