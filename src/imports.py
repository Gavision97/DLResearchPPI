import os
import time
import copy
import warnings
warnings.filterwarnings('ignore')

from typing import List, Tuple, Dict

import sys
import logging

# Data manipulation
import pandas as pd
import numpy as np

# Scikit-learn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import resample

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.ML.Cluster import Butina

# Deep learning: PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import Dataset, DataLoader, Subset

# Hugging Face Transformers (for ChemBERTa)
from transformers import (
    RobertaTokenizer, RobertaModel, RobertaConfig, 
    AdamW, get_linear_schedule_with_warmup, BertModel
)

# Chemprop and DeepChem
import chemprop
from chemprop import data, featurizers, models
import deepchem as dc

# Abstract Base Classes (for defining abstract methods)
from abc import ABC, abstractmethod