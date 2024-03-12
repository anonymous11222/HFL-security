import numpy as np 
from art.estimators.classification import KerasClassifier

def create_Preprocessing_defence (defence_name):
  
  ###Preprocessing
  if defence_name == 'SpatialSmoothing':
    from art.defences.preprocessor import SpatialSmoothing
    defence_instant = SpatialSmoothing(window_size=7)
  if defence_name == 'GaussianAugmentation':
    from art.defences.preprocessor import GaussianAugmentation
    defence_instant = GaussianAugmentation(sigma=9, augmentation=False)
  if defence_name == 'FeatureSqueezing':
    from art.defences.preprocessor import FeatureSqueezing
    defence_instant = FeatureSqueezing(bit_depth=3, clip_values=(0, 255))
  if defence_name == 'TotalVarMin':
    from art.defences.preprocessor import TotalVarMin
    defence_instant = TotalVarMin(norm=1)
  if defence_name == 'CutMix': #****
    from art.defences.preprocessor import CutMix
    defence_instant = CutMix(num_classes=10, alpha=1, probability=1,apply_predict=True, apply_fit=False)
  if defence_name == 'Cutout': 
    from art.defences.preprocessor import Cutout
    defence_instant = Cutout(length=4,apply_predict=True, apply_fit=False)
  if defence_name == 'JpegCompression': 
    from art.defences.preprocessor import JpegCompression
    defence_instant = JpegCompression(clip_values=(0, 1))
  return defence_instant   

def create_postprocessing_defence (defence_name):    
   ####postprocessing
  if defence_name == 'ReverseSigmoid': 
    from art.defences.postprocessor import ReverseSigmoid
    defence_instant = ReverseSigmoid(beta=1.0,gamma=0.2)
  if defence_name == 'GaussianNoise': 
    from art.defences.postprocessor import GaussianNoise
    defence_instant = GaussianNoise()        
  if defence_name == 'HighConfidence': 
    from art.defences.postprocessor import HighConfidence
    defence_instant = HighConfidence(cutoff =0.25)
  if defence_name == 'Rounded': 
    from art.defences.postprocessor import Rounded
    defence_instant = Rounded(decimals =3)
  return defence_instant    

def create_detector_defence (defence_name,classifier, detector,data):  
  ###Detection
  if defence_name == 'BinaryActivationDetector':
    from art.defences.detector.evasion import BinaryActivationDetector
    defence_instant = BinaryActivationDetector(classifier=classifier, detector=detector, layer=0)
  if defence_name == 'BinaryInputDetector':
    from art.defences.detector.evasion import BinaryInputDetector
    defence_instant = BinaryInputDetector(detector)
  if defence_name == 'SubsetScanningDetector':
    from art.defences.detector.evasion import SubsetScanningDetector
    defence_instant = SubsetScanningDetector(classifier, bgd_data=data, layer=1)                       
             
   
  return defence_instant
  
def create_transformer_defence (defence_name,trained_classifier,batch_size,nb_epochs):  
  ###Detection
  if defence_name == 'DefensiveDistillation':
    from art.defences.transformer.evasion import DefensiveDistillation
    defence_instant = DefensiveDistillation(classifier=trained_classifier, batch_size=batch_size, nb_epochs=nb_epochs)
              
   
  return defence_instant    
  
def get_preprocessed_data(defence_name,x_to_be_processed, y_to_be_processed):
    defence_instant = create_Preprocessing_defence (defence_name)
    x_def,_= defence_instant(x_to_be_processed)
    x_processed = np.array(x_def)
    return  x_processed,y_to_be_processed   
    
def get_postprocessing_protected_model(defence_name,uprotected_model):
    defence_instant = create_postprocessing_defence (defence_name)
    classifier_with_defence = KerasClassifier(model=uprotected_model, clip_values=(0, 1),postprocessing_defences=defence_instant)
    return  classifier_with_defence  
    
def get_detector_model(defence_name,classifier , detector, x_train_detector, y_train_detector, nb_epochs=2, batch_size=128):
    defence_instant = create_detector_defence (defence_name,classifier, detector,x_train_detector)
    if defence_name != 'SubsetScanningDetector':
      defence_instant.fit(x_train_detector, y_train_detector, nb_epochs=nb_epochs, batch_size=batch_size)
    return  defence_instant
    
def get_transformed_model(defence_name,trained_classifier,transformed_classifier,train_data,batch_size,nb_epochs):
    transformed_classifier = KerasClassifier(model=transformed_classifier, clip_values=(0, 1))
    defence_instant = create_transformer_defence (defence_name,trained_classifier,batch_size,nb_epochs)
    transformed_classifier = defence_instant(x=train_data, transformed_classifier=transformed_classifier)
    return  transformed_classifier        
