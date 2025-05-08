#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import csv
import re
import json
import os 
import argparse
import numpy as np


def get_last_iteration():
    """
    Reads from progress file and finds the iteration number
    or zero if their isn't one in the file.
    
    """
    if os.path.exists("progress.txt"):
        with open("progress.txt", "r") as f:
            return int(f.read().strip())
    return 0

def read_csv_excluding_special_chars(file_path, encoding='utf-8'):
    """
    Reads a CSV file and returns its contents as a list of rows, 
    excluding special characters from each cell.
    """
    data = []
    with open(file_path, 'r', encoding=encoding) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            cleaned_row = [re.sub(r'[^a-zA-Z0-9\s]', '', cell) for cell in row]
            #should truncate the  passsages

            data.append(cleaned_row)
    return data

# Example usage:
file_path = 'train_subset.csv'
data = read_csv_excluding_special_chars(file_path)


# Set up argument parser
parser = argparse.ArgumentParser(description="Resume from last iteration if needed.")
parser.add_argument("--resume", type=int, default=get_last_iteration(), help="Iteration to resume from")
args = parser.parse_args()



tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
pipe = pipeline(
    model="google/gemma-2-9b-it",
)
batch_size = 1;
with open("output.jsonl", "w", encoding="utf-8") as f:
    for i in range(args.resume, len(data), batch_size):
        #list of rows specified in the batchsize
        batch = data[i:i+batch_size]
        #this creates a list of prompts with the right text content using the list conprehension to
        #make a certain amount of the given f_string.
        #todo:makes prompt small
        prompts = [
            f"extract claim from text: {row[0][:300]}"
            for row in batch
        ]
        

        
        # tokens = tokenizer.tokenize(text= f"{messages}")
        #set max tokens to limit the length of answers.

        generated_claim = pipe(prompts, return_full_text=False, max_new_tokens = 10)

        for j, output in enumerate(generated_claim):
            
            #should acsses the specific generation in the multiple responses generated
            generated_text = output[j]['generated_text']
            # answer stripping and adding to list
            if "Claim:" in generated_text:
                generated_text = generated_text.split("**Claim:**")[-1].strip()
            data[i + j].append(generated_text)
            
            # Evaluation of answer
            #batch is the chunck of the lists in data that contain passages and reference claims
            reference = batch[j][1]
            candidate = generated_text
            tokenized_reference = word_tokenize(reference)
            tokenized_candidate = word_tokenize(candidate)
            score = meteor_score([tokenized_reference], tokenized_candidate)
            data[i+j].append(score)
            json.dump(data[i + j], f, ensure_ascii=False)
            # Separate each JSON object with a newline
            f.write("\n")  
        
        # saves progress
        with open("progress.txt", "w") as progress_file:
            progress_file.write(str(i))



# List to store the values to average
scores = []

# Read the JSONL file line by line
with open("output.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        obj = json.loads(line.strip())
          # Parse each line as a JSON object
        #might need something to check if that index exists, since an error could make the object not have three elements.
        if isinstance(obj, list) and len(obj) > 3:
            scores.append(obj[3])

# Compute the average and adds them to a new file
if scores:
    average_score = sum(scores) / len(scores)
    standard_deviation = np.std(scores, ddof = 1)
    data = {
        "Mean": average_score,
        "Standard deviation": standard_deviation 
    }
    print(f"Average score: {average_score:.2f}")
    print(f"std: {standard_deviation:.2f}")
    with open("stats.jsonl", "w", encoding="utf-8") as file:
        json.dump(data, file)
        
else:
    print("No valid scores found.")


# Data is a list of lists that each contain a passage at [0], the corrrect reference prompt [1], the generated answer [2], and the meteor score [3]. 

