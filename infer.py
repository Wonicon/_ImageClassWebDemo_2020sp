import sys
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import os
import cv2

from vgg19 import VGG
import resnet

VGG_MODEL = None
RESNET_MODEL = None


def vgg(img, device):
  global VGG_MODEL
  if not VGG_MODEL:
    print("vgg init")
    checkpoint = torch.load('model.pth', map_location=torch.device(device))
    model = VGG('VGG19')
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)
    model.eval()
    VGG_MODEL = model

  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  img = transform_test(img)
  img = img.to(device)
  img = img.unsqueeze(0)
  output = VGG_MODEL(img)
  prob = F.softmax(output,dim=1) 
  value, predicted = torch.max(output.data, 1)
  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  pred_class = classes[predicted.item()]
  return pred_class


def resnet32(img, device):
  global RESNET_MODEL
  if not RESNET_MODEL:
    print("resnet init")
    model = torch.nn.DataParallel(resnet.__dict__['resnet32']())
    checkpoint = torch.load('save_temp/checkpoint.th', map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    RESNET_MODEL = model


  trans = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])

  img = trans(img)
  img = img.to(device)
  img = img.unsqueeze(0)
  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  with torch.no_grad():
    output = RESNET_MODEL(img)
    prob = F.softmax(output,dim=1) 
    value, predicted = torch.max(output.data, 1)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    pred_class = classes[predicted.item()]
    return pred_class


def infer(imgfile, models):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    img = cv2.imread(imgfile)
    img = cv2.resize(img, (32, 32))

    result = []
    if 'resnet' in models:
      result.append(resnet32(img, device))
    if 'vgg' in models:
      result.append(vgg(img, device))
    return result
