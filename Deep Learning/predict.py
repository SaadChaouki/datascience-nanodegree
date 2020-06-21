
import functions
import torch
import numpy as np
import json
import argparse
import sys


if __name__ == '__main__':
    # Creating parser for training
    Arguments = sys.argv[1:]
    Parser = functions.ArgumentParsers('predict')
    ParsedArguments = Parser.parse_args(Arguments)
    
    # Processing Image
    ProcessedImage = functions.process_image(ParsedArguments.path_to_image)

    # Loading Model
    LoadedModel = functions.Load(ParsedArguments.checkpoint_path)

    # Predicting
    Probs, Classes = functions.predict(ProcessedImage, LoadedModel, topk = ParsedArguments.top_k)
   
    # Mapping the classes to names
    FlowerNames = []
    if ParsedArguments.category_name != None:
        try:
            with open('cat_to_name.json', 'r') as f:
                cat_to_name = json.load(f)
            for Class in Classes:
                FlowerNames.append(cat_to_name[Class])
        except:
            print("Mapping File not found.")

    print("-"*50)
    print("Predictions:")
    print(FlowerNames if len(FlowerNames) > 0 else Classes)
    print(Probs)
    print("-"*50)
