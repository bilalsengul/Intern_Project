import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image
import cv2, os
from sklearn.preprocessing import OneHotEncoder

import torch, torchvision
import torchvision.transforms as transforms
from torchvision import datasets as dset
from torch.utils.data import DataLoader

json_path = "../data/json/js"
data_path = "../data/image/jpg"

def get_filenames(path):
  return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def get_jsonnames(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.json')]

file_names = get_filenames(data_path)
file_names[:5]


json_names = get_jsonnames(json_path)
json_names[:5]

print(json_names)


def equalize_hist(img):
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return cv2.equalizeHist(gray_img)

def equalize_histogram(json):
  gray_json = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
  return cv2.equalizeHist(gray_json)

for infile in file_names:
  # Read image
  img = cv2.imread(infile)
  #print(img)
  # Check if it's a three channel image
  if (len(img.shape)== 3):
    outfile = equalize_hist(img)
    
    cv2.imwrite(infile, outfile)
   
  else:
    print('No RGB image')

for injfile in json_names:
  # Read image
  json = cv2.imread(infile)
  outfile = equalize_histogram(img)
  cv2.imwrite(infile, outfile)   
     


    



def loader(data_dir, img_size, batchSize):
  # Preprocessing: Resize, brightness corrections
  dataset = dset.ImageFolder(root = data_dir,
                            transform=transforms.Compose([
                            transforms.Resize(img_size),
                           
                            transforms.ToTensor(),
                            ]))
  dataloader = torch.utils.data.DataLoader(dataset,
                                          batch_size= batchSize,
                                          shuffle=True)
  
  print('Data size:', len(dataset), 'images')

  
  return dataloader

def jsonloader(data_dir, img_size, batchSize):
  # Preprocessing: Resize, brightness corrections
  dataset = dset.DatasetFolder(root = data_dir,loader=None,
                            transform=transforms.Compose([
                            transforms.Resize(img_size),
                           
                            transforms.ToTensor(),
                            ]))
  dataloader = torch.utils.data.DataLoader(dataset,
                                          batch_size= batchSize,
                                          shuffle=True)
  
  print('Data size:', len(dataset), 'json')

  
  return dataloader


data_path = "../data/image"
json_path = "../data/json"
json_path_out = "../data"
sample_loader = loader(data_path, (200,200), 15) 
json_loader = jsonloader(json_path, (200,200), 8,json_path_out) 

#onehot_encoder = OneHotEncoder(sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#print(onehot_encoded)


sample_iter = iter(sample_loader)
json_iter = iter(json_loader)

sample, labels  = sample_iter.next()
json  = json_iter.next()
print('Sample and images shape on BatchSize = {}'.format(sample.size()))
