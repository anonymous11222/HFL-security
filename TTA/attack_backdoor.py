# Backdoor-blackbox attack functions


from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.utils import to_categorical
import numpy as np

def poison_dataset(
  clean_images,
  clean_labels,
  target_labels,
  source_labels,
  n_classes,
  percent_poison
  ):
  # Creating copies of our clean images and labels
  # Poisoned samples will be added to these copies
  x_poison = clean_images.copy()
  y_poison = clean_labels.copy()
    
    
  # Array to indicate if a sample is poisoned or not
  # 0s are for clean samples, 1s are for poisoned samples
  is_poison = np.zeros(shape=y_poison.shape[0])


  # Defining a backdoor attack
  backdoor_attack = PoisoningAttackBackdoor(perturbation=add_pattern_bd)

  # Iterating over our source labels and provided target labels
  for (source_label, target_label) in (zip(source_labels, target_labels)):
      # Calculating the number of clean labels that are equal to the
      # current source label
      num_labels = np.size(np.where(np.argmax(a=clean_labels, axis=1) == source_label))

      # Calculating the number of samples that should be poisoned from
      # the current source labels
      num_poison = round(percent_poison * num_labels)
      print("num_poison: " +str(num_poison))
      # Getting the images for the current clean label
      source_images = clean_images[np.argmax(a=clean_labels, axis=1) == source_label]
      print("Be-source_images.shape: " +str(source_images.shape))  
      ##########
      source_images_idx = np.where(np.argmax(a=clean_labels, axis=1) == source_label)
      source_lables_ = clean_labels[source_images_idx]

      x_poison = np.delete(x_poison, source_images_idx, axis=0)
      y_poison = np.delete(y_poison, source_images_idx, axis=0)

      ###########
      
      # Randomly picking indices to poison
      indices_to_be_poisoned = np.random.choice(
          a=num_labels,
          size=num_poison,
          replace=False
          )

      # Get the images for the current label that should be poisoned
      images_to_be_poisoned = source_images[indices_to_be_poisoned].copy()

      ########  
      source_images = np.delete(source_images, indices_to_be_poisoned, axis=0)
      source_lables_ = np.delete(source_lables_, indices_to_be_poisoned, axis=0) 

      #########  
        
      # Converting the target label to a categorical
      target_label = to_categorical(labels=(np.ones(shape=num_poison) * target_label), nb_classes=n_classes)

      # Poisoning the images and labels for the current label
      poisoned_images, poisoned_labels = backdoor_attack.poison(
          x=images_to_be_poisoned,
          y=target_label
          )
      print("poisoned_images.shape: " +str(poisoned_images.shape)) 
      # Appending the poisoned images to our clean images
      x_poison = np.append(
          arr=x_poison,
          values=poisoned_images,
          axis=0
          )

      # Appending the poisoned labels to our clean labels
      y_poison = np.append(
          arr=y_poison,
          values=poisoned_labels,
          axis=0
          )
      print("Af-source_images.shape: " +str(source_images.shape))  
      ##########################     
      # Appending the poisoned images to our clean images
      x_poison = np.append(
          arr=x_poison,
          values=source_images,
          axis=0
          )

      # Appending the poisoned labels to our clean labels
      y_poison = np.append(
          arr=y_poison,
          values=source_lables_,
          axis=0
          ) 

    #############################    
      # Appending 1s to the poison indicator array
      is_poison = np.append(
          arr=is_poison,
          values=np.ones(shape=num_poison)
          )
  print("is_poison.shape: " +str(is_poison.shape))
  print("x_poison.shape: " +str(x_poison.shape))
  # Returning the poisoned samples and the poison indicator array
  return is_poison, x_poison, y_poison
def generate(train_images, train_labels, target_labels, source_labels,  n_classes, percent_poison):

  #percent_poison = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80 , 0.9 , 1.0]

  (is_poison_train, poison_train_images, poison_train_labels) = poison_dataset(
      clean_images=train_images,
      clean_labels=train_labels,
      target_labels=target_labels,
      source_labels=source_labels,
      n_classes = n_classes,
      percent_poison=percent_poison)

  # Shuffling the training data
  num_train = poison_train_images.shape[0]
  shuffled_indices = np.arange(num_train)
  np.random.shuffle(shuffled_indices)
  poison_train_images = poison_train_images[shuffled_indices]
  poison_train_labels = poison_train_labels[shuffled_indices]
  is_poison_train = is_poison_train[shuffled_indices]

  return is_poison_train, poison_train_images, poison_train_labels
  
  