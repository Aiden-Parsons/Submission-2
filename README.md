# Submission-2
Submission-2 code for CS881

Author: Aiden Parsons To run my method of claim generation and evaluation, I would recommend making sure your python version is up to date before anything else. That might not matter too much if you are using an IDE like Pycharm or VScode that stay generally up to date. I use a Hugging Face transformers library to run one of their models on my laptop/server. This does require you to make an acount on Hugging Face, and get and activate a personal API key to run it. 

The page with the model on it: https://huggingface.co/google/gemma-2-9b-it and agreeing to googles terms and conditions as well as the API key.
The dependencies to pip install in your python environment are pytorch, transformers, and nltk. For example: 'pip install -U transformers' would work for transformers and https://www.nltk.org/install.html has a guide on the installations for what device you're running on. This page is a tutorial on how to install pytorch: https://pytorch.org/get-started/locally/

I will look into pyproject.toml in the furture and hopefully update this once I have further looked into how to set it up.

Index for files:

The Gemma3.py file on this repository is the code that I use to do everything including, extract from the csv, prompting the model, evlauating the response, writing the output,and then evaluating the output.

The CSV files are the needed training and dev set to evaluate on.
