import numpy as np 
from art.estimators.classification import KerasClassifier

  
def create_detector_defence (defence_name,classifier, x_data,y_data):  
  ###Detection
  if defence_name == 'ActivationDefence':
    from art.defences.detector.poison import ActivationDefence
    defence_instant = ActivationDefence(classifier=classifier, x_train=x_data, y_train=y_data)
  if defence_name == 'ProvenanceDefense':
    from art.defences.detector.poison import ProvenanceDefense
    defence_instant = ProvenanceDefense(classifier=classifier, x_train=x_data, y_train=y_data)
  if defence_name == 'RONIDefense':
    from art.defences.detector.poison import RONIDefense
    defence_instant = RONIDefense(classifier=classifier, x_train=x_data, y_train=y_data)                  
             
   
  return defence_instant
  
def create_transformer_defence (defence_name,trained_classifier):  

  if 'NeuralCleanse' in defence_name:
    from art.defences.transformer.poisoning import NeuralCleanse
    defence_instant = NeuralCleanse(classifier=trained_classifier)
    
  if 'STRIP' in defence_name:
    from art.defences.transformer.poisoning import STRIP
    defence_instant = STRIP(classifier=trained_classifier)            
   
  return defence_instant    
  

    
def get_detector_report(defence_name,model , x_data,y_data):
    classifier = KerasClassifier(model=model, clip_values=(0, 1))
    defence_instant = create_detector_defence (defence_name,classifier, x_data,y_data)
    report, is_clean_reported = defence_instant.detect_poison(nb_clusters=2, reduce="PCA", nb_dims=10)
    return  is_clean_reported
    
def get_transformed_model(defence_name,trained_classifier,x_data,y_data):
    trained_classifier = KerasClassifier(model=trained_classifier, clip_values=(0, 1))
    defence_instant = create_transformer_defence (defence_name,trained_classifier)
    if 'NeuralCleanse' in defence_name:
        transformed_classifier  = defence_instant(trained_classifier, steps=10, learning_rate=0.1)
        mitigation_types = []
        if 'filtering' in defence_name:
            mitigation_types.append("filtering")
        if 'unlearning' in defence_name:
            mitigation_types.append("unlearning")
        if 'pruning' in defence_name:
            mitigation_types.append("pruning")
        transformed_classifier.mitigate(x_data,y_data, mitigation_types=mitigation_types) 
    if 'STRIP' in defence_name:           
        transformed_classifier  = defence_instant()
        transformed_classifier.mitigate(x_data)
    return  transformed_classifier        
