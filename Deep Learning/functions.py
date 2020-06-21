import argparse
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
import torch

def GetDataLoaders(data_dir, train = '/train', valid = '/valid', test = '/test'):
    # setting train, validation, and testing directories.
    train_dir = data_dir + train
    valid_dir = data_dir + valid
    test_dir = data_dir + test
    
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224), 
                                           transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # torchvision.transforms.ColorJitter
    # torchvision.transforms.RandomApply(transforms, p=0.5)

    test_valid_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Loading dataset. Adding try to handle cases when folder structure is not recognised or folder does not exist.
    try:
        train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
        valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
        test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)
    except:
        print('Folders structure not recognised.')
        return None

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    # Combining data loaders in dict.
    dataloaders = {'train': trainloader, 'valid': validloader, 'test':testloader }
    
    print("Data Loader created.")
    
    return dataloaders, train_data.class_to_idx

def Evaluate(Model, DataLoader, Criterion, device):
    Model.to(device)
    Model.eval()
    EvaluationLoss = 0
    EvaluationAccuracy = 0
    for images, labels in DataLoader:
        images, labels = images.to(device), labels.to(device)
        
        Predictions = Model.forward(images)
        EvaluationLoss += Criterion(Predictions, labels).item()
        
        ExpPreds = torch.exp(Predictions)
        EqualityChecks = (labels.data == ExpPreds.max(dim=1)[1])
        EvaluationAccuracy += EqualityChecks.type(torch.FloatTensor).mean()
    return EvaluationLoss, EvaluationAccuracy

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    try:
        img = Image.open(image)
    except:
        print('Image not found.')
        return None
    
    # Resizing the image to 256x256
    img = img.resize((256, 256), resample = 0)

    # Crop 224*224 center
    width, height = img.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    # Crop the center of the image
    CroppedImg = img.crop((left, top, right, bottom))
    
    # Transforming the image to an array to standardize and normalize
    NumpyImage = np.array(CroppedImg)
    NumpyImage = NumpyImage/255
    NumpyImage = (NumpyImage - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    NumpyImage = NumpyImage.transpose((2, 0, 1))
    ProcessedImage = torch.from_numpy(NumpyImage)
    return ProcessedImage
    
def ArgumentParsers(train_predict):
    if train_predict == 'train':
        parser = argparse.ArgumentParser(description='Parsing Training Arguments')
        parser.add_argument('data_directory')
        parser.add_argument('--gpu', action = 'store_true')
        parser.add_argument('--verbose', default = 'store_false')
        parser.add_argument('--arch', default = 'vgg11')
        parser.add_argument('--save_dir', default = '')
        parser.add_argument('--learning_rate', default = 0.001, type=float)
        parser.add_argument('--hidden_units', nargs='+', default = [4096], type=int)
        parser.add_argument('--epochs', default = 20, type=int)
    elif train_predict == 'predict':
        parser = argparse.ArgumentParser(description='Parsing Predicting Arguments')
        parser.add_argument('path_to_image')
        parser.add_argument('checkpoint_path')
        parser.add_argument('--gpu', action = 'store_true')
        parser.add_argument('--top_k', default = 5, type = int)
        parser.add_argument('--category_name')
    return parser    
    
def GetPreTrainedModel(PreTrainedModel):
    '''
    Function to return a pretrained model from the pytorch models. The exec function is used so that the user
    case choose from all the available models without being restricted to the hard coded ones.
    '''
    exec('global model; model = models.' + PreTrainedModel + '(pretrained=True)')
    global model
    for param in model.parameters():
        param.requires_grad = False
    return model

def BuildClassifier(Layers = [25088, 4096, 102], Activations = ['relu', 'logsoftmax'], Architecture = 'vgg11'):
    '''
        Function to build a sequential classifier. Created to use a dynamic combination of activation functions.
        NLayers = Number of layers in the classifier including input and output.
        Layers = Size of each layer. The first and last numbers refer to the input and output sizes respectively.
        Activations = Activation function to be used after each layer.
    '''
    # Check data is correct.
    ImplementedActivations = set(['relu', 'logsoftmax', 'sigmoid'])
    if len(set(Activations) - ImplementedActivations) > 0:
        print('Activation Functions not recognised.')
        return None
    
    ClassDict = OrderedDict()
    for i in range(len(Layers) - 1):
        # Adding the layer
        ClassDict['Lay_' + str(i)] = nn.Linear(in_features = Layers[i], out_features = Layers[i + 1])
        # Activation Function
        if Activations[i] == 'relu':
            ClassDict['Act_' + str(i)] = nn.ReLU()
        elif Activations[i] == 'logsoftmax':
            ClassDict['Act_' + str(i)] = nn.LogSoftmax(dim=1)
        elif Activations[i] == 'sigmoid':
            ClassDict['Act_' + str(i)] = nn.Sigmoid()

    # Build Classifier
    BuiltClassifier = nn.Sequential(ClassDict)
    
    # Getting Architecture Model
    ArchModel = GetPreTrainedModel(Architecture)
    ArchModel.classifier = BuiltClassifier
    
    print("Model Built.")
    
    return ArchModel

def Load(Path):
    try:
        ModelInformation = torch.load(Path)
    except:
        print("Model Not Found")
        return None
    # Build model.
    Model = BuildClassifier(ModelInformation['layers'], ModelInformation['activations'],ModelInformation['trained_model'])
    Model.load_state_dict(ModelInformation['state_dict'])
    Model.class_to_idx = ModelInformation['class_to_idx']
    print("Model Successfuly loaded.")
    return Model

def Train(Model, DataLoaders, Optimizer, Criterion,  epochs = 5, gpu = True, Verbose = True, VSteps = 5):
    '''
        Function that controls the overall process of getting the data, creating the model, and training it. 
    '''
    # Checking device
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    print("Training on " + ("cuda." if torch.cuda.is_available() and gpu else "cpu."))
    
    print("Initating Model Training.")
    
    # Setting the model to the device used.
    Model.to(device)
    steps = 0
    # Training
    for epoch in range(epochs):
        Model.train()
        print("Starting Epoch:" + str(epoch + 1))
        running_loss = 0
        for inputs, labels in DataLoaders['train']:
            steps += 1
            
            # Changing the  inputs to the device used.
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Setting gradients to 0
            Optimizer.zero_grad()

            # Forward and backward passes
            outputs = Model.forward(inputs)
            loss = Criterion(outputs, labels)
            loss.backward()
            Optimizer.step()
            
            # Computing Loss
            running_loss += loss.item()
            
            # Verbose
            if Verbose and steps % VSteps == 0:
                with torch.no_grad():
                    ValLoss, ValAcc = Evaluate(Model, DataLoaders['valid'], Criterion, device)
                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/VSteps),
                      "Validation Loss: {:.3f}.. ".format(ValLoss/len(DataLoaders['valid'])),
                      "Validation Accuracy: {:.3f}".format(ValAcc/len(DataLoaders['valid'])))
                running_loss = 0
                Model.train()

    print("Model Trained.")
    return Model
  
def Save(Model, Path, PreTrainedModel, SizeLayers, Activations):
    Model.cpu()
    ModelInformation = {'trained_model': PreTrainedModel,
                'layers': SizeLayers,
                'activations': Activations,
                'state_dict': Model.state_dict(),
                'class_to_idx': Model.class_to_idx}

    # Saving the model
    torch.save(ModelInformation, Path)
    print("Model Saved Succesfuly.")

    
def predict(ProcessedImage, Model, topk=5, gpu = True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    
    Model.to(device)
    ProcessedImage = ProcessedImage.type(torch.FloatTensor).unsqueeze_(0).to(device)
    Predictions = Model.forward(ProcessedImage)
    TransformedPrediction = torch.exp(Predictions)
    Probabilities, Indices = TransformedPrediction.topk(topk) 
    
    # Get classes instead of indices
    Index_To_Class = {index: label for label, index in Model.class_to_idx.items()}
    
    Probabilities = Probabilities[0].cpu().data.numpy()
    Indices = Indices[0].cpu().data.numpy()
    
    # Classes
    Classes = []
    for Index in Indices:
        Classes.append(Index_To_Class[Index])

    return Probabilities, Classes