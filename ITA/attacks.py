import numpy as np
from art.attacks.evasion import AdversarialPatch
def  Adversarial_Patch(x_to_be_poisoned,y_to_be_poisoned,classifier):
  ap = AdversarialPatch(classifier=classifier, max_iter=2,targeted =False)
  patch, patch_mask = ap.generate(x=x_to_be_poisoned, y=y_to_be_poisoned)
  poisoned_x = ap.apply_patch(x_to_be_poisoned, scale=0.5)
  return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import AutoProjectedGradientDescent
def  AutoPGD(x_to_be_poisoned,y_to_be_poisoned,classifier):
  attack = AutoProjectedGradientDescent(estimator=classifier, max_iter=2, eps=0.5)
  poisoned_x = attack.generate(x=x_to_be_poisoned, y=y_to_be_poisoned)
  return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import AutoConjugateGradient
def  AutoCG(x_to_be_poisoned,y_to_be_poisoned,classifier):
  attack = AutoConjugateGradient(estimator=classifier, max_iter=2, eps=0.5)
  poisoned_x = attack.generate(x=x_to_be_poisoned, y=y_to_be_poisoned)
  return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import BoundaryAttack
def  Boundary_Attack(x_to_be_poisoned,y_to_be_poisoned,classifier):
  attack = BoundaryAttack(estimator=classifier, max_iter=2, epsilon=0.5)
  poisoned_x = attack.generate(x=x_to_be_poisoned, y=y_to_be_poisoned)
  return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import CarliniL0Method
def  CarliniL0_Method(x_to_be_poisoned,y_to_be_poisoned,classifier):
  attack = CarliniL0Method(classifier=classifier, max_iter=2, targeted =False, batch_size= 32, verbose=True)
  poisoned_x = attack.generate(x=x_to_be_poisoned, y=y_to_be_poisoned)
  return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import CarliniL2Method
def  CarliniL2_Method(x_to_be_poisoned,y_to_be_poisoned,classifier):
  attack = CarliniL2Method(classifier=classifier, max_iter=2, targeted =False, batch_size= 32,confidence=1)
  poisoned_x = attack.generate(x=x_to_be_poisoned, y=y_to_be_poisoned)
  return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import CarliniLInfMethod
def  CarliniLInf_Method(x_to_be_poisoned,y_to_be_poisoned,classifier):
  attack = CarliniLInfMethod(classifier=classifier, max_iter=2, targeted =False, batch_size= 32)
  poisoned_x = attack.generate(x=x_to_be_poisoned, y=y_to_be_poisoned)
  return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import DeepFool
def  Deep_Fool(x_to_be_poisoned,y_to_be_poisoned,classifier):
  attack = DeepFool(classifier=classifier, max_iter=2, epsilon=0.5, batch_size= 32)
  poisoned_x = attack.generate(x=x_to_be_poisoned, y=y_to_be_poisoned)
  return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import ElasticNet
def Elastic_Net(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = ElasticNet(classifier=classifier, targeted= False, max_iter=2, batch_size= 32)
    poisoned_x = attack.generate(x=x_to_be_poisoned, y=y_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import FastGradientMethod
def fgm(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = FastGradientMethod(estimator=classifier, eps=0.5)
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import FeatureAdversariesNumpy
def Feature_AdversariesNumpy(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = FeatureAdversariesNumpy(classifier=classifier, batch_size= 32, delta=0.5, layer=2)
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import FrameSaliencyAttack
def Frame_SaliencyAttack(x_to_be_poisoned,y_to_be_poisoned,classifier):
    a = FastGradientMethod(estimator=classifier, eps=0.5)
    attack = FrameSaliencyAttack(classifier=classifier, batch_size= 32, attacker=a)
    poisoned_x = attack.generate(x=x_to_be_poisoned, y=y_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import GeoDA
def Geo_DA(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = GeoDA(estimator=classifier, batch_size= 64)
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import GRAPHITEBlackbox
def GRAPHITE_Blackbox(x_to_be_poisoned,y_to_be_poisoned,classifier):
    #init_image = x_to_be_poisoned[1, :, :, :]
    target_image = x_to_be_poisoned[3, :, :, :]
    #attack = GRAPHITEBlackbox(classifier, (32, 32), (32, 32), (4, 4), batch_size = 1, num_xforms_mask = 50, num_xforms_boost = 50, heatmap_mode = 'Target', num_boost_queries = 200 * 50, blur_kernels = [1], tr_lo = 0.4, tr_hi = 0.6)
    #x_adv = attack.generate(x=x_test[1, :, :, :][np.newaxis, :, :, :], y=y_test[3][np.newaxis, :], mask=None, x_test[3, :, :, :][np.newaxis, :, :, :])_
    attack = GRAPHITEBlackbox(classifier=classifier, batch_size= 64, noise_size=dataset_shape, net_size =dataset_shape,num_xforms_mask = 50, num_xforms_boost = 50, heatmap_mode = 'Target', num_boost_queries = 200 , blur_kernels = [1], tr_lo = 0.4, tr_hi = 0.6,)
    poisoned_x = attack.generate(x=x_to_be_poisoned, y=y_to_be_poisoned, x_tar=target_image )
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import HopSkipJump
def Hop_SkipJump(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = HopSkipJump(classifier=classifier, batch_size= 64, max_iter=2)
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import ProjectedGradientDescentNumpy
def pgd(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = ProjectedGradientDescentNumpy(estimator=classifier, eps=0.5, max_iter=2, batch_size= 64)
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import LaserAttack
def Laser_Attack(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = LaserAttack(estimator=classifier, iterations=2)
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import LowProFool
def Low_ProFool(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = LowProFool(classifier =classifier, importance = 'pearson')
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import NewtonFool
def Newton_Fool(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = NewtonFool(classifier=classifier, batch_size= 64, max_iter=2)
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import PixelAttack
def Pixel_Attack(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = PixelAttack(classifier=classifier, max_iter=2,verbose=True)
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned


from art.attacks.evasion import ThresholdAttack
def Threshold_Attack(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = ThresholdAttack(classifier=classifier, max_iter=2)
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned


from art.attacks.evasion import SaliencyMapMethod
def Saliency_MapMethod(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = SaliencyMapMethod(classifier=classifier, batch_size=64)
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import SignOPTAttack
def SignOPT(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack_SignOPTAttack = SignOPTAttack(estimator=classifier, batch_size=64,targeted= False, epsilon=0.001, max_iter=2, query_limit= 10, verbose=True)
    poisoned_x = attack_SignOPTAttack.generate(x=x_to_be_poisoned, y=y_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import SimBA
def Sim_BA(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack= SimBA(classifier=classifier, targeted= False, epsilon=0.5, max_iter=3000,attack="px")
    poisoned_x = attack.generate(x=x_to_be_poisoned, y=y_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import SpatialTransformation
def Spatial_Transformation(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack= SpatialTransformation(classifier=classifier,max_translation = 50, num_translations = 4, max_rotation = 45, num_rotations= 4 )
    poisoned_x = attack.generate(x=x_to_be_poisoned, y=y_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned


from art.attacks.evasion import SquareAttack
def Square_Attack(x_to_be_poisoned,y_to_be_poisoned,classifier):
    squareAttack = SquareAttack(estimator =classifier, eps=0.5)
    poisoned_x = squareAttack.generate(x=x_to_be_poisoned,max_iter=2 )
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import TargetedUniversalPerturbation
def Targeted_UniversalPerturbation(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = TargetedUniversalPerturbation(classifier=classifier, eps=0.5, max_iter=2)
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import VirtualAdversarialMethod
def Targeted_UniversalPerturbatio(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = TargetedUniversalPerturbation(classifier=classifier, eps=0.5, max_iter=2,batch_size=64 )
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned

from art.attacks.evasion import Wasserstein
def Wasser_stein(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = Wasserstein(estimator =classifier, eps=0.5, max_iter=2,batch_size=64 )
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned


from art.attacks.evasion import ZooAttack
def Zoo_Attack(x_to_be_poisoned,y_to_be_poisoned,classifier):
    attack = ZooAttack(classifier =classifier, max_iter=2,batch_size=64 )
    poisoned_x = attack.generate(x=x_to_be_poisoned)
    return poisoned_x,y_to_be_poisoned
    
def part_poisoned(x_clean,y_clean,poisoned_x,poisoned_y,percent_poison):
  num_poison = round(percent_poison * len(x_clean))
  indices_to_be_poisoned = np.random.choice(
            a=range(len(x_clean)),
            size=num_poison,
            replace=False)

  x_to_be_poisoned = poisoned_x[indices_to_be_poisoned].copy()
  y_to_be_poisoned = poisoned_y[indices_to_be_poisoned].copy()
  x_clean = np.delete(x_clean, indices_to_be_poisoned, axis=0)
  y_clean = np.delete(y_clean, indices_to_be_poisoned, axis=0)

  return x_to_be_poisoned, y_to_be_poisoned, x_clean, y_clean

def to_be_poisoned(x_clean,y_clean,percent_poison):
  num_poison = round(percent_poison * len(x_clean))
  indices_to_be_poisoned = np.random.choice(
            a=range(len(x_clean)),
            size=num_poison,
            replace=False)
  x_to_be_poisoned = x_clean[indices_to_be_poisoned].copy()
  y_to_be_poisoned = y_clean[indices_to_be_poisoned].copy()
  x_clean = np.delete(x_clean, indices_to_be_poisoned, axis=0)
  y_clean = np.delete(y_clean, indices_to_be_poisoned, axis=0)

  return x_to_be_poisoned, y_to_be_poisoned, x_clean, y_clean

import tensorflow as tf
from art.estimators.classification import KerasClassifier
tf.compat.v1.disable_eager_execution()

def generate_poisoned_data(x_to_be_poisoned, y_to_be_poisoned,model,attack_name):
    if 'defence' in attack_name:
      attack_name = attack_name.replace('defence', '')
      classifier = model
    else:
      classifier = KerasClassifier(model=model, clip_values=(0, 1))
    if attack_name == 'Adversarial_Patch':
          poisoned_x,poisoned_y = Adversarial_Patch(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'AutoPGD':
          poisoned_x,poisoned_y = AutoPGD(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'AutoCG':
          poisoned_x,poisoned_y = AutoCG(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Boundary_Attack':
          poisoned_x,poisoned_y = Boundary_Attack(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'CarliniL0_Method':
          poisoned_x,poisoned_y = CarliniL0_Method(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'CarliniL&Wagner L2': #'CarliniL2_Method':
          poisoned_x,poisoned_y = CarliniL2_Method(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'CarliniL&Wagner Linf':
          poisoned_x,poisoned_y = CarliniLInf_Method(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Deep_Fool':
          poisoned_x,poisoned_y = Deep_Fool(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Elastic_Net':
          poisoned_x,poisoned_y = Elastic_Net(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'fgm':
          poisoned_x,poisoned_y = fgm(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Feature_AdversariesNumpy':
          poisoned_x,poisoned_y = Feature_AdversariesNumpy(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Frame_SaliencyAttack':
          poisoned_x,poisoned_y = Frame_SaliencyAttack(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Geo_DA':
          poisoned_x,poisoned_y = Geo_DA(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'GRAPHITE_Blackbox':
          poisoned_x,poisoned_y = GRAPHITE_Blackbox(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Hop_SkipJump':
          poisoned_x,poisoned_y = Hop_SkipJump(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'pgd':
          poisoned_x,poisoned_y = pgd(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Laser_Attack':
          poisoned_x,poisoned_y = Laser_Attack(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Low_ProFool':
          poisoned_x,poisoned_y = Low_ProFool(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Newton_Fool':
          poisoned_x,poisoned_y = Newton_Fool(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Pixel_Attack':
          poisoned_x,poisoned_y = Pixel_Attack(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Threshold_Attack':
          poisoned_x,poisoned_y = Threshold_Attack(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Saliency_Map':
          poisoned_x,poisoned_y = Saliency_MapMethod(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'SignOPT':
          poisoned_x,poisoned_y = SignOPT(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Sim_BA':
          poisoned_x,poisoned_y = Sim_BA(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Spatial_Transformation':
          poisoned_x,poisoned_y = Spatial_Transformation(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Square_Attack':
          poisoned_x,poisoned_y = Square_Attack(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Targeted_UniversalPerturbation':
          poisoned_x,poisoned_y = Targeted_UniversalPerturbation(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Wasserstein':
          poisoned_x,poisoned_y = Wasser_stein(x_to_be_poisoned,y_to_be_poisoned,classifier)
    if attack_name == 'Zoo_Attack':
          poisoned_x,poisoned_y = Zoo_Attack(x_to_be_poisoned,y_to_be_poisoned,classifier)

    return  poisoned_x,poisoned_y
    
