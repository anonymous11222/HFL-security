import argparse
import os.path
import math
import numpy as np
import random
import time
import pickle
from os import path
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, losses
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


from TTA import attack_lableflipping, attack_backdoor
from ITA import attacks


parser = argparse.ArgumentParser()
parser.add_argument('-config', '--configurations', help='experiment configurations')
prompt_args = parser.parse_args()


import tomli
with open(prompt_args.configurations, mode="rb") as fp:
  args = tomli.load(fp)
 
###### Experiment configuration ######

exp_id = args["experimnet_id"]
saved_data_path = args["data_path"]
work_space = args["store_folder"]
dataset_name = args["dataset_name"]

if dataset_name == 'cifar10':
  n_classes , image_shape = 10 , (32, 32, 3)
elif dataset_name == 'mnist':  
  n_classes , image_shape = 10 , (28, 28, 1)
elif dataset_name == 'fashion-mnist':
  n_classes , image_shape = 10 , (28, 28, 1)


#topology
num_clients = args["topology_description"]["number_of_clients"]
topology_levels = args["topology_description"]["topology_levels"]
topology = args["topology"]
topology_name = args["topology_description"]["topology_name"]

#local training
epochs = args["local_training"]["epochs"]
batch_size = args["local_training"]["batch_size"]

#fedrated training
aggregation_round = args["federated_training"]["aggregation_round"]
percent_of_participats = args["federated_training"]["percent_of_participats"]
start_training = args["federated_training"]["start_training"]
if start_training == 'start':
  initial_training = True
  continual_training = False
  start_from = 1
elif start_training == 'resume':  
  initial_training = False
  continual_training = True
  continual_training_path = args["federated_training"]["resume_from_model"]
  start_from = args["federated_training"]["resume_from_round_number"]
  training_time = args["federated_training"]["resume_from_training_time"]

defence_name = 'none'
#distillation_training
distillation_enable =  args["distillation_training"]["distillation_enable"]
if distillation_enable:
  teacher_model_path = args["distillation_training"]["teacher_model"]
  defence_name = 'distillation'
  
#adversarial_training
adversarial_training_enable =  args["adversarial_training"]["adversarial_training_enable"]
if adversarial_training_enable:
  continual_training_path = args["federated_training"]["resume_from_model"]
  defence_name =  'adversarial_training'
#Attack configuration
attack_enable = args["attack"]["attack_enable"] 


if attack_enable or adversarial_training_enable:
  DPA_list = ['backdoor','lable_flipping']
  MPA_list = ['signflip']
  AT_list = ['fgm','Spatial_Transformation']
  attack_name = args["attack"]["attack_name"] 
  adv_nodes = args["attack"]["malicious_nodes"] 
  num_malicious_nodes = len(adv_nodes)
  if attack_name in  DPA_list or AT_list:
    DPA_enable = True
    MPA_enable = False
    percent_poison = args["attack"]["percent_of_poison_data"]
    source_labels=np.arange(n_classes)
    target_labels=(np.arange(n_classes) + 1) % n_classes
  
  if attack_name in  MPA_list:
    MPA_enable = True
    DPA_enable = False
 


else:
 
  attack_name ='none'
  num_malicious_nodes = 0
  adv_nodes = [] 
  
###### End of Experiment configuration ######

os.chdir(work_space)
exp_main_folder_path = work_space 
training_time = datetime.now().strftime("%d-%m-%y_%I-%M-%S-%p") 
print('Experiment start at: ' + str(training_time))
exp_description = topology_name + '_' + str(topology_levels) + 'L_' + dataset_name + '_' + str(num_clients) + '_attack_' + attack_name + str(num_malicious_nodes) + '_defence_'+ defence_name
exp_folder = exp_main_folder_path + '/Exp_' + str(exp_id) + '_' + training_time + '_' + exp_description
os.mkdir(exp_folder)
for l in range(topology_levels):
  os.mkdir(exp_folder + '/level'+str(l))


def mk_folder(parent_folder, child_folder):
    if not os.path.exists(exp_folder + '/' + parent_folder + '/' + child_folder):
      os.mkdir(exp_folder + '/' + parent_folder + '/' + child_folder)
      os.mkdir(exp_folder + '/' + parent_folder + '/' + child_folder + '/memory')
      os.mkdir(exp_folder + '/' + parent_folder + '/' + child_folder + '/memory/data')
      os.mkdir(exp_folder + '/' + parent_folder + '/' + child_folder + '/memory/model')

level=0
#Output folders of the experiment
def topology_walk(t,level):
    for key, value in t.items():
        if type(value) == dict:  
            print(key)
            print(level)  # this line is for printing each nested key
            mk_folder('level'+str(level), key)
            level=level+1
            print(level)
            topology_walk(value,level)
        else:

           # print(level)
            mk_folder('level'+str(level), key)
            level=level+1
            for i in value:
              mk_folder('level'+str(level), i)
              print(key, ': ', i)
        level = level - 1
        #print(level)

topology_walk(topology, 0)
global_round = 1
os.chdir(exp_folder)


# Training models

def tf_create_model():
    # Defining the CNN model
    if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
   
      model = Sequential()
      model.add(Conv2D(32, (3, 3), padding="same", input_shape=image_shape))
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Conv2D(64, (3, 3),padding='same'))
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Flatten())
      model.add(Dense(512))
      model.add(Activation("relu"))
      model.add(Dropout(0.5))
      model.add(Dense(10))
      model.add(Activation("softmax"))

    if dataset_name == 'cifar10' or dataset_name == 'stl10':


      model = Sequential()
      model.add(Conv2D(32, (3, 3), padding="same", input_shape=image_shape))
      model.add(Activation("relu"))
      model.add(Conv2D(32, (3, 3)))
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Dropout(0.25))

      model.add(Conv2D(64, (3, 3), padding="same"))
      model.add(Activation("relu"))
      model.add(Conv2D(64, (3, 3)))
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Dropout(0.25))

      model.add(Flatten())
      model.add(Dense(512))
      model.add(Activation("relu"))
      model.add(Dropout(0.5))
      model.add(Dense(10))
      model.add(Activation("softmax"))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate= 3e-4,
    decay_steps=32,
    decay_rate=0.1)

    # Compiling the model
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics= ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

    # Returning the model
    return model

#from keras.optimizers.legacy import Adam
m = tf_create_model()



class Node(object):

      def __init__(self, id, name, node_type, node_level):
        self.id = id
        self.name = name
        self.node_type = node_type
        self.node_level = node_level
        self.memory_path = 'level' + str(self.node_level) + '/' + self.name + '/memory'
        self.data_path = saved_data_path + '/' + self.name + '/clean'
        self.global_model= '' #path
        self.local_model= '' #path
        self.local_model_num = 0
        self.training_epoch = 10
        self.training_batch_size= 32



      def set_id(self, id):
          self.id = id
      def set_name(self, name):
          self.name = name
      def set_node_type(self, node_type):
          self.node_type = node_type
      def set_node_level(self, node_level):
          self.node_level = node_level
      def set_memory_path(self, memory_path):
          self.memory_path = memory_path
      def set_global_model(self, model):
          self.global_model = model
      def set_local_model(self, model):
          self.local_model = model
      def set_training_epoch(self, training_epoch):
          self.training_epoch = training_epoch
      def set_training_batch_size(self, training_batch_size):
          self.training_batch_size= training_batch_size


      def get_id(self):
          return self.id
      def get_name(self):
          return self.name
      def get_node_type(self):
          return self.node_type
      def get_node_level(self):
          return self.node_level
      def get_memory_path(self):
          return self.memory_path
      def get_global_model(self):
          return self.global_model
      def get_local_model(self):
          return self.local_model
      def get_training_epoch(self):
          return self.training_epoch
      def get_training_batch_size(self):
          return self.training_batch_size


      def print_info (self):
          print('&&&&& Id: ' + str(self.get_id()))
          print('&&&&& Name: ' + self.get_name())
          print('&&&&& Type: ' + self.get_node_type())
          print('&&&&& Level: ' + str(self.get_node_level()))
          print('&&&&& Memory path: ' + self.get_memory_path())
          print('&&&&& Global model: ' + self.get_global_model())
          print('&&&&& Local model: ' + self.get_local_model())
          print('&&&&& training epoch: ' + str(self.get_training_epoch()))
          print('&&&&& training batch_size: ' + str(self.get_training_batch_size()))


      def get_clean_data(self):
          train_images = np.load(self.data_path + '/x_train.npy')
          train_labels = np.load(self.data_path + '/y_train.npy')
          num_samples = len(train_images)
          return (train_images, train_labels,  num_samples)


      def upload_model(self):
          return self.get_local_model()

      def download_model(self, model):
          self.set_global_model(model)

      def local_training(self, inital_training= True):
          if inital_training:
              model = tf_create_model()
              print('&&&&&&&&&&&&&&&&&&&&&& START OF INITAL TRAINING &&&&&&&&&&&&&&&&&&&')
          else:
              model = tf.keras.models.load_model(self.global_model)  # load a pretrained model (recommended for training)
              print('&&&&&&&&&&&&&&&&&&&&&& START OF LOCAL TRAINING &&&&&&&&&&&&&&&&&&&')
          (train_images, train_labels, num_samples) = self.get_clean_data()
          self.print_info ()
          history = model.fit(x=train_images, y=train_labels, epochs=self.training_epoch, batch_size=self.training_batch_size)
          model_path = self.memory_path + '/model/local_model'+'.h5'
          model.save(model_path)#, save_format='tf')
          print('&&&&&&&&&&&&&&&&&&&&&& Finish Local Training &&&&&&&&&&&&&&&&&&&&' )
          self.set_local_model(model_path)
          return  self.get_local_model(),num_samples
     
class Client(Node):

      def __init__(self, id, name, node_type, node_level):
        super().__init__(id, name, node_type, node_level)
        self.data_path = saved_data_path + '/' + self.name + '/clean'
        self.DPA_enable = False
        self.MPA_enable = False
        self.train_images = []
        self.train_labels = []
        self.num_samples = 0

      def set_data_path(self, data_path):
          self.data_path = data_path
      def set_DPA_enable(self, DPA_enable):
          self.DPA_enable = DPA_enable
      def set_MPA_enable(self, MPA_enable):
          self.MPA_enable = MPA_enable
      def set_train_images(self, train_images):
          self.train_images = train_images
      def set_train_labels(self, train_labels):
          self.train_labels = train_labels        

      def get_data_path(self):
          return self.data_path
      def get_DPA_enable(self):
          return self.DPA_enable
      def get_MPA_enable(self):
          return self.MPA_enable

      def print_info (self):
          super().print_info ()
          print('&&&&& Data path: ' + self.get_data_path())
          print('&&&&& DPA enable: ' + str(self.get_DPA_enable()))
          print('&&&&& MPA enable: ' + str(self.get_MPA_enable()))


      def get_poisoned_data(self,train_images, train_labels):
         
          if attack_name == 'backdoor':
            
            (is_poison_train, poison_train_images, poison_train_labels) = attack_backdoor.generate(train_images, train_labels,  target_labels, source_labels,n_classes, percent_poison)
            num_samples = len(poison_train_images)
            
          if attack_name == 'lable_flipping':
        
            (poison_train_images, poison_train_labels) = attack_lableflipping.generate(train_images, train_labels)
            num_samples = len(poison_train_images)
            
          if adversarial_training_enable:
           

            model = tf.keras.models.load_model(continual_training_path)
            (poison_train_images, poison_train_labels) = attacks.generate_poisoned_data(train_images, train_labels,model,attack_name)
            poison_train_images = np.concatenate((poison_train_images, train_images), axis=0)
            poison_train_labels = np.concatenate((poison_train_labels, train_labels), axis=0)
            num_samples = len(poison_train_images)
              
          
          return (poison_train_images, poison_train_labels,  num_samples)

      def get_poisoned_model(self,model):
          if attack_name == 'signflip':
            params = model.get_weights()
            temp = np.array(params)
            temp = temp * -1
            p = temp.tolist()
            model.set_weights(p)
          return (model)

      def local_training(self, inital_training= True,distillation_enable=distillation_enable):
        if inital_training:
            model = tf_create_model()
            print('&&&&&&&&&&&&&&&&&&&&&& START OF INITAL TRAINING &&&&&&&&&&&&&&&&&&&')
        else:
            model = tf.keras.models.load_model(self.global_model)  # load a pretrained model (recommended for training)
            print('&&&&&&&&&&&&&&&&&&&&&& START OF LOCAL TRAINING &&&&&&&&&&&&&&&&&&&')
            
        self.print_info ()
     
              
        if distillation_enable:
      
           teacher_model = tf.keras.models.load_model(teacher_model_path)
           preds = teacher_model.predict(x=self.train_images, batch_size=self.training_batch_size)
           self.set_train_labels(preds)
           distillation_enable = False
           
       
        history = model.fit(x=self.train_images, y=self.train_labels, epochs=self.training_epoch, batch_size=self.training_batch_size)

        if self.MPA_enable:
          model = self.get_poisoned_model(model)
   

        model_path = self.memory_path + '/model/local_model' +'.h5'
        model.save(model_path)
        print('&&&&&&&&&&&&&&&&&&&&&& Finish Local Training &&&&&&&&&&&&&&&&&&&&' )
   
        self.set_local_model(model_path)
        return  self.get_local_model(),  self.num_samples 
        
class Server(Node):

      def __init__(self, id, name, node_type, node_level):
          super().__init__(id, name, node_type, node_level)
          self.data_path = saved_data_path + '/client0/clean'
          self.local_model_num = 1
          self.child_nodes = []
          self.aggregation_round = 1
          self.child_nodes_per_round = len(self.child_nodes)
          self.local_model_num = 1
          self.global_model_num = 1
          self.MPA_enable = False

      def set_data_path(self, data_path):
          self.data_path = data_path
      def set_child_node(self, child_node):
          self.child_nodes.append(child_node)
      def set_aggregation_round(self, aggregation_round):
          self.aggregation_round = aggregation_round
      def set_child_nodes_per_round(self, child_nodes_per_round):
          self.child_nodes_per_round = child_nodes_per_round
      def set_MPA_enable(self, MPA_enable):
          self.MPA_enable = MPA_enable    


      def get_data_path(self):
          return self.data_path
      def get_child_nodes(self):
          return self.child_nodes
      def get_aggregation_round(self):
          return self.aggregation_round
      def get_child_nodes_per_round(self):
          return self.child_nodes_per_round
      def get_MPA_enable(self):
          return self.MPA_enable    

      def get_participants_per_round(self,nonparticipant):
          partcipant = []
         
          if len(nonparticipant) < self.get_child_nodes_per_round() or len(nonparticipant) < self.aggregation_round:
            nonparticipant = self.child_nodes.copy()
          partcipant = random.sample(nonparticipant, self.get_child_nodes_per_round())
          for item in partcipant:
            nonparticipant.remove(item)
          return partcipant,nonparticipant

      def print_info (self):
          super().print_info()
          print('&&&&& Data path: ' + self.get_data_path())
          print('&&&&& Number of child nodes: '  + str(len(self.get_child_nodes())))
          print('&&&&& Aggregation_round: ' + str(self.get_aggregation_round()))
          print('&&&&& MPA enable: ' + str(self.get_MPA_enable()))

      def get_poisoned_model(self,model):
          if attack_name == 'signflip':
            params = model.get_weights()
            temp = np.array(params)
            temp = temp * -1
            p = temp.tolist()
            model.set_weights(p)
          return (model)    

      def aggregation (self, praticipant, server_round, global_round):
          #print(self.node_level)
          w_locals = []
          model_path = self.global_model #edge global model
          model_last = tf.keras.models.load_model(model_path)
          params_last = model_last.get_weights()
          array1 = np.array(params_last)
          sum_num_samples = 0

          #print(len(praticipant))
          for p in praticipant:

              print('>>>>>>>>>>>>>>>>>>>>>>>>>>')
              print('>>> 3- Model Distribuion: send edge global model to ' + p.name + ' >>>')
              p.download_model(self.global_model)
              print('>>> Global model sent! >>>')
              print('>>>>>>>>>>>>>>>>>>>>>>>>>>')

              print('>>>>>>>>>>>>>>>>>>>>>>>>>>')
              print('>>> 4- local training: Train local model of ' + p.name + ' >>>')
              if self.node_level == level-1:
                
                model_path, num_samples = p.local_training(inital_training= False)

              else:
               
                model_path, num_samples = p.start_aggregation(inital_training = False, continual_training =False, global_round=global_round)

              model_update = tf.keras.models.load_model(model_path)
              print('>>> Local training Done! >>>')
              print('>>>>>>>>>>>>>>>>>>>>>>>>>>')


              params = model_update.get_weights()
              array2 = np.array(params)
              array1 = array1 + (num_samples* array2)
              sum_num_samples = sum_num_samples + num_samples
      
              K.clear_session()

          '''for ind in range(len(array1)):
            array1[ind]/=sum_num_samples #len(praticipant)'''

          array1/=sum_num_samples
       

          print('>>>>>>>>>>>>>>>>>>>>>>>>>>')
          print('>>> 5- Model aggregation: aggregate parameters from all clients models>>>')

          del(params_last)
          print('>>>Model aggregation Done! >>>')
          print('>>>>>>>>>>>>>>>>>>>>>>>>>>')
          global_model_path = self.global_model #edge global model
          print(model_path)
          model_update = tf.keras.models.load_model(model_path)
      
          model_update.set_weights(array1)
          if self.MPA_enable:
            model_update = self.get_poisoned_model(model_update)
          
          updated_model_path = self.memory_path + '/model/global_model_round_'+ str(global_round) +'.h5'
          self.global_model_num += 1
          model_update.save(updated_model_path)
          self.set_global_model(updated_model_path)
          return self.global_model, sum_num_samples

      def start_aggregation (self, inital_training = True, continual_training =False, global_round=global_round):

        print('******************** START OF FEDERATED LEARNING of SERVER ' + self.name + ' *****')
        self.print_info()
        start_time = datetime.now()
        t = start_time.strftime("%H:%M:%S")
        print('***** Starting time: ' + t )
        round_counter = 1
        if inital_training:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('>>> 1- Initial global model training: train initial global model in server>>>')
            local_model, num_samples = self.local_training(inital_training)
            self.set_global_model(local_model)

            print('>>> Initial global model training Done>>>')
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>')

        if continual_training:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('>>> 1- continual edge global model training: train initial global model in servers>>>')
            local_model = continual_training_path
            num_samples = 550 #any number
            self.set_global_model(local_model)
            round_counter = start_from
            print('>>> continual edge global model training Done>>>')
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>')

        inital_training = False
        continual_training = False

        if self.node_level == topology_levels-2:
          self.set_child_nodes_per_round (math.ceil (len(self.get_child_nodes()) * percent_of_participats))
        print('number:'+ str(percent_of_participats) + '_' + str(len(self.get_child_nodes())) + '_' + str(self.child_nodes_per_round))

        nonparticipant = self.get_child_nodes().copy()
        while round_counter <= self.get_aggregation_round():
            if self.node_level == 0:
              global_round = round_counter
            print('*************************************')
            print('***** Start round: %r*****' %round_counter )
            print('***** Global model: ' + self.global_model)
            print('***** Local model: ' + self.local_model)
            print('*************************************')
            if self.node_level == topology_levels - 2:
              participants, nonparticipant = self.get_participants_per_round(nonparticipant)
              print('here2:'+ str(self.node_level) + ' ' + str(len(participants)))
            else:
              participants = self.get_child_nodes().copy()
              print('here:'+ str(self.node_level) + ' ' + str(len(participants)))
          
            model, num_samples = self.aggregation(participants, round_counter, global_round)
            print('*************************************')
            print('***** Finish iteration: %r*****' %round_counter)
            print('*************************************')
            round_counter += 1

        end_time = datetime.now()
        t = end_time.strftime("%H:%M:%S")
        
        print('*************************************')
        print('***** End time: ' + t )
        print('******************** End OF FEDERATED LEARNING of server ' + self.name + '*****')
       
        model = tf.keras.models.load_model(self.global_model)
      
        model_path = self.memory_path + '/model/local_model_round_'+ str(self.local_model_num) +'.h5'
        self.local_model_num +=1
        model.save(model_path)
        self.set_local_model(model_path)
        return self.local_model, num_samples

#*********************************************************************************************************
       
nodes = []
for level in range (topology_levels):
  nodes.append([])

def topology_walk2(t,level,id):
    for parent, childs in t.items():

        if type(childs) == dict:
          nodes[level].append(Server(id=id, name = parent, node_type = 'server', node_level = level))
          id = id + 1
          level = level + 1
          topology_walk2(childs,level,id)
        else:
          nodes[level].append(Server(id=id, name = parent, node_type = 'server', node_level = level))
          id = id + 1
          level = level + 1
          for child in childs:
         
            nodes[level].append(Client(id=id, name = child, node_type = 'client', node_level = level))
            id = id + 1
        
          
        level = level - 1

topology_walk2(topology,0,0)


for level in range (topology_levels):
  for node in range(len(nodes[level])):
    print(nodes[level][node].name)
    
for level in range (topology_levels):
  for node in range(len(nodes[level])):

    nodes[level][node].set_training_epoch(epochs)
    nodes[level][node].set_training_batch_size(batch_size)
    if level != topology_levels-1 :
      nodes[level][node].set_aggregation_round(aggregation_round[level])
    if level == topology_levels-2:
       for adv_node in adv_nodes:
        if nodes[level][node].name == adv_node:
          if MPA_enable:
            nodes[level][node].set_MPA_enable(MPA_enable) 
    if level == topology_levels-1:
      (train_images, train_labels,  num_samples) = nodes[level][node].get_clean_data()
      for adv_node in adv_nodes:
        if nodes[level][node].name == adv_node:
          if DPA_enable:
            nodes[level][node].set_DPA_enable(DPA_enable)
            (train_images, train_labels,  num_samples) = nodes[level][node].get_poisoned_data(train_images, train_labels)
          elif MPA_enable:
            nodes[level][node].set_MPA_enable(MPA_enable) 
             
      nodes[level][node].train_images = train_images
      nodes[level][node].train_labels = train_labels
      nodes[level][node].num_samples = num_samples
                    

level = 0
def topology_walk3(t,level):
    parent_inx = 0
    child_inx = 0
    for parent, childs in t.items():
      print('parent:' + parent )
      for node in range(len(nodes[level])):
          if nodes[level][node].name == parent:
            parent_inx = node
      for child in childs:
        print('child:' + child)
        for node in range(len(nodes[level+1])):
            if nodes[level+1][node].name == child:
              child_inx = node
        nodes[level][parent_inx].set_child_node(nodes[level+1][child_inx])
      if type(childs) == dict:
          topology_walk3(childs,level+1)


topology_walk3(topology,0)

for level in range (topology_levels):
  for node in range(len(nodes[level])):
    print(nodes[level][node].print_info())
    
start_time = time.time()

nodes[0][0].start_aggregation(inital_training = initial_training, continual_training =continual_training, global_round=global_round)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
                                    