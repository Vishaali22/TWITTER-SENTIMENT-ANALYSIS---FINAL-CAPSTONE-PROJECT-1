# TWITTER-SENTIMENT-ANALYSIS---FINAL-CAPSTONE-PROJECT-1
TWITTER SENTIMENT ANALYSIS - FINAL CAPSTONE PROJECT 1
---Synopsis
->This Python script performs text preprocessing, vectorization, model training, and evaluation on a subset of tweets from a dataset to predict the toxicity of tweets. It uses several machine learning models and visualizes their learning curves to compare performance.

--Imports necessary libraries:
->pandas for data manipulation
->numpy for numerical operations
->matplotlib and seaborn for plotting
->re and unicodedata for text preprocessing
->wordcloud for generating word cloud images
->nltk for natural language processing tasks
->sklearn for machine learning models and evaluation metrics

--Loads the dataset from a CSV file and takes a random sample of 1000 rows for faster processing and testing.

--Defines a function to preprocess text:
->Converts text to lowercase
->Removes emails, URLs, special characters, and accented characters

--Applies the preprocessing function to the 'tweet' column in the dataframe.
--Defines the feature set (X) and labels (y) for the machine learning models.
--Splits the data into training and testing sets with an 80-20 split, ensuring stratification to maintain the same distribution of classes in both sets.
--Defines and applies a TF-IDF vectorizer to convert text data into numerical features, limiting to 1000 features and removing English stop words.
--Defines a dictionary of models to be evaluated: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and Support Vector Classifier (SVC).
--Defines a function to plot learning curves for a given model, showing the training and cross-validation scores as a function of the number of training examples.
--Generates and displays learning curves for each model in the models dictionary.
--Trains each model, makes predictions, and prints evaluation metrics: accuracy, classification report, and confusion matrix.

--Generate Word Cloud for Toxic Tweets
->Creates and displays a word cloud from the toxic tweets in the dataset to visualize the most common words in toxic tweets.

---Conclusion
->This script demonstrates a comprehensive approach to text classification using various machine learning models. It includes steps for data preprocessing, feature extraction, model training and evaluation, and visualization of learning curves and word clouds. By running this script, one can compare the performance of different models on the task of classifying tweets based on their toxicity.















