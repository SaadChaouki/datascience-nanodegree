# Libraries

import functions
import torch
import numpy as np
import argparse
from torch import nn
from torch import optim
import sys


if __name__ == '__main__':
    # Fixed Parameters
    InputSize = 25088
    OutputSize = 102
    
    # Creating parser for training
    Arguments = sys.argv[1:]
    Parser = functions.ArgumentParsers('train')
    ParsedArguments = Parser.parse_args(Arguments)
    
    # Building Layers of Model. The number of hidden units can be several layers.
    ModelLayers = [InputSize]
    ModelActivations = []

    for HiddenUnit in ParsedArguments.hidden_units:
        ModelLayers.append(HiddenUnit)
        ModelActivations.append('relu')
  
    ModelLayers.append(OutputSize)
    ModelActivations.append('logsoftmax')

    # Creating Data Loaders
    DataLoaders, class_to_idx = functions.GetDataLoaders(ParsedArguments.data_directory)
   
    # Creating model
    Model = functions.BuildClassifier(Layers = ModelLayers, Activations = ModelActivations, Architecture = ParsedArguments.arch.lower())
    
    # Optimizer
    Optimizer = optim.Adam(Model.classifier.parameters(), lr=ParsedArguments.learning_rate)
    
    # Criterion
    Criterion = nn.NLLLoss()

    # Training the model
    Model = functions.Train(Model, DataLoaders, Optimizer, Criterion, epochs = ParsedArguments.epochs, gpu = ParsedArguments.gpu)
       
    # Adding Classes to model
    Model.class_to_idx = class_to_idx

    # Save the trained model if a path was given.
    if ParsedArguments.save_dir != '':
        functions.Save(Model, ParsedArguments.save_dir, ParsedArguments.arch.lower(), ModelLayers, ModelActivations)
    
    

