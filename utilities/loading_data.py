import numpy as np
import torch
from bisect import bisect
import os
from torch.utils.data import Dataset, DataLoader

def to_numpy(x):
  return x.detach().cpu().numpy()

#files Loader
def MyLoader(GL, do = "train", config = None, args=None):
  if config is not None: 
    batch_size  = config['train']['batchsize']
    workers = config['data']['load_workers']
    database = config['Project']['database']
    if database == 'GRF_7Hz': 
      size = 128
    elif database in {'GRF_12Hz','GRF_15Hz'}: 
      size = 64
  elif args is not None:  
    batch_size = args.batchsize
    workers = args.load_workers
    database = args.database
    if database == 'GRF_7Hz': 
      size = 128
    elif database in {'GRF_12Hz', 'GRF_15Hz'}: 
      size = 64
  else: 
    batch_size = 50
    workers = 4


  if do == 'train': 
    list_x_train, list_y_train = GL('train')
    list_x_valid, list_y_valid = GL('validation')
    Train_Data_set = File_Loader(list_x_train,list_y_train, size = size, data=database)
    Valid_Data_set = File_Loader(list_x_valid,list_y_valid, size = size, data=database)
    ##### setting the data Loader
    train_loader = DataLoader(dataset = Train_Data_set, 
                         shuffle = True, 
                         batch_size = batch_size,
                         num_workers= workers)

    valid_loader = DataLoader(dataset = Valid_Data_set, 
                            shuffle = False, 
                            batch_size =batch_size,
                            num_workers= workers)
    return train_loader, valid_loader

  elif do == 'test':
    list_x_test, list_y_test = GL('test')
    Test_Data_set = File_Loader(list_x_test, list_y_test, size = size, data=database)
    ##### setting the data Loader
    test_loader = DataLoader(dataset = Test_Data_set, 
                            shuffle = False, 
                            batch_size = batch_size,
                            num_workers= workers)
    return test_loader

class GettingLists(object):
  #Generating the list for train/valid/test--> each sample is 5000 velocity/data
  def __init__(self, data_for_training, 
                    wave_eq = "acoustic",
                    data_base = "GRF_7Hz", 
                    PATH = 'databases', 
                    batch_data_size = int(5000)):
    super(GettingLists, self).__init__()
    self.PATH = os.path.join(PATH, wave_eq, data_base)
    self.batch_data = batch_data_size
    valid_limit = data_for_training//self.batch_data 
    self.valid_limit = valid_limit
    if data_base == 'GRF_7Hz': 
      self.end = int(6) 
    elif data_base in {'GRF_12Hz', 'GRF_15Hz'} :
      self.end = int(10)
   
  def get_list(self, do):
    if do == 'train':
      in_limit_train  = np.array([os.path.join(self.PATH, 
                                                'model', 
                                                f'velocity{k}.npy') for k in \
                                                range(1,self.valid_limit+1)])
      out_limit_train = np.array([os.path.join(self.PATH, 
                                                'data', 
                                                f'pressure{k}.npy')for k in \
                                                range(1,self.valid_limit+1)])
      return in_limit_train, out_limit_train


    elif do == 'validation':
      in_limit_valid  = np.array([os.path.join(self.PATH, 
                                              'model', 
                                              f'velocity{k}.npy') for k in \
                                              range(self.end,self.end+1)])
      out_limit_valid= np.array([os.path.join(self.PATH,
                                              'data', 
                                              f'pressure{k}.npy') for k in \
                                              range(self.end,self.end+1)])
      return  in_limit_valid, out_limit_valid

    elif do =='test':
      in_limit_test  = np.array([os.path.join(self.PATH,
                                               'model', f'velocity{k}.npy') for k in \
                                                range(self.valid_limit+1, self.end+1)])
      out_limit_test = np.array([os.path.join(self.PATH, 
                                                'data', f'pressure{k}.npy')for k in \
                                                range(self.valid_limit+1, self.end+1)])
      return in_limit_test, out_limit_test
  
  def __call__(self, do = 'train'):
    return self.get_list(do)


class File_Loader(Dataset):
    #data loader file
    def __init__(self, data_paths, target_paths, size =128, data = "GRF"):
        self.size = size
        self.data = data

        if self.data == "GRF_7Hz":
          self.data_memmaps = [np.load(path, mmap_mode='r') for path in data_paths]
          self.target_memmaps = [np.load(path, mmap_mode='r') for path in target_paths]
        elif self.data == ("GRF_12Hz") or ("GRF_15Hz") :
          self.data_memmaps = [np.load(path, mmap_mode='r').view(float) for path in data_paths]
          self.target_memmaps = [np.load(path, mmap_mode='r').view(float) for path in target_paths]
        elif self.data == ("GRF_12Hz_vz") or ("GRF_15Hz_vz"):
          self.data_memmaps = [np.load(path, mmap_mode='r').view(float) for path in data_paths]
          self.target_memmaps = [np.load(path, mmap_mode='r').view(float).reshape(2,self.size,self.size,2) for path in target_paths]
  
        self.start_indices = [0] * len(data_paths)
        self.data_count = 0

        for index, memmap in enumerate(self.data_memmaps):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        data = np.copy(self.data_memmaps[memmap_index][index_in_memmap])
        target = np.copy(self.target_memmaps[memmap_index][index_in_memmap])
        if self.data == "GRF_7Hz":
          return torch.tensor(data*1e-3, dtype=torch.float).view(self.size,self.size,1), torch.tensor(target, dtype=torch.float).view(self.size,self.size,1)
        elif self.data == ("GRF_12Hz") or ("GRF_15Hz"):
          return torch.tensor(data*1e-3, dtype=torch.float).view(self.size,self.size,1), torch.tensor(target, dtype=torch.float).view(self.size,self.size,2)
        elif self.data == ("GRF_12Hz_vz") or ("GRF_15Hz_vz"):
          return torch.tensor(data*1e-3, dtype=torch.float).view(self.size,self.size,1), torch.tensor(target, dtype=torch.float).view(2,self.size,self.size,2)
