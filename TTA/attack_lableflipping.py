import random
import numpy as np
def generate(train_images, train_labels):
    
 
  random.seed()
  poison_train_labels = np.random.permutation(train_labels)
  poison_train_labels = np.random.permutation(poison_train_labels)
  
  return train_images, poison_train_labels