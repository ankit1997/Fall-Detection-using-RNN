import os
import numpy as np
from tqdm import tqdm
from scipy import signal

num_sensors = 3
TIME_STEPS = 450

class Data:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        num_classes = len(np.unique(self.labels))
        
        # convert labels to one-hot
        temp = np.zeros((self.labels.shape[0], num_classes))
        temp[range(self.labels.shape[0]), self.labels] = 1
        self.labels = temp
        
        # shuffle data
        self.shuffle()
    
    def shuffle(self):
        # np.random.seed(13)
        np.random.seed(17)
        indices = np.array(range(self.labels.shape[0]))
        np.random.shuffle(indices)
        self.features = self.features[indices]
        self.labels = self.labels[indices]

class DataLoader:
    '''
    Loads data from MobiFall_Dataset_v2.0/ directory.
    The result is stored in the dictionary.
    '''
    
    def __init__(self, path=''):
        self.path = path
        self.ADL_TYPES = ["CSI", "CSO", "JOG", "JUM", "SCH", "STD", "STN", "STU", "WAL"]
        self.FALL_TYPES = ["BSC", "FKL", "FOL", "SDL"]
        self.classes = ["CSI", "CSO", "JOG", "JUM", "SCH", "STD", "STN", "STU", "WAL", "BSC", "FKL", "FOL", "SDL"]
        self.dataset = {}
        self._load_data()
        self._format_dataset()
    
    def _load_data(self):
        '''
        Load the dataset, starting from root directory `self.path`.
        '''
        print("Loading dataset...")
        subdirs = self._get_subdirs()
        
        ADLs = []
        FALLs = []
        for i in range(len(subdirs)):
            subdir = subdirs[i]
            class_dirs = self._get_subdirs(subdir)
            for class_dir in class_dirs:
                if class_dir.endswith('ADL'):
                    ADLs.append(class_dir)
                elif class_dir.endswith('FALLS'):
                    FALLs.append(class_dir)
                else:
                    print("Found unusual directory:", class_dir)
        self._process_category(ADLs, self.ADL_TYPES)
        self._process_category(FALLs, self.FALL_TYPES)
    
    def _process_category(self, dirs=[], category_types=[]):
        '''
        Process ADL/FALL directories to load data.
        '''
        for i in tqdm(range(len(dirs))):
            dir_ = dirs[i]
            subdirs = self._get_subdirs(dir_)
            assert len(subdirs) == len(category_types), "Found an unusual class in " + dir_
            
            for subdir in subdirs:
                class_ = self._get_class(subdir)
                data = self._prepare_dataset_from_dir(subdir)
                
                if data.shape[0] == 0:
                    # MobiFall_Dataset_v2.0/sub21/FALLS/SDL is empty; deal with this here.
                    continue
                
                try:
                    self.dataset[class_] = np.concatenate((self.dataset[class_], data), axis=0)
                except KeyError:
                    self.dataset[class_] = data
    
    def _prepare_dataset_from_dir(self, folder=''):
        '''
        Prepares dataset from .txt files within `folder`.
        '''
        files = self._get_trails(folder)
        num_trails = len(files)
        num_trails_per_sensor = num_trails // num_sensors
        
        data = []
        for i in range(num_trails_per_sensor):
            data_files = []
            for s in range(num_sensors):
                data_files.append(files[i+s*num_trails_per_sensor])
            data.append(self._accumulate(data_files))
        
        return np.array(data)
    
    def _accumulate(self, files=[]):
        '''
        Args:
            files = A list of files containing individual sensor readings
        Return:
            Accumulated results as a numpy array
        '''
        assert len(files) >= 1
        features = self._read_file(files[0], interpolate=True)
        for i in range(1, len(files)):
            f = self._read_file(files[1], interpolate=True)
            features = np.concatenate((features, f), axis=1)
        return features
    
    def _read_file(self, fname='', interpolate=False):
        assert fname != ''
        data = ''
        with open(fname) as file:
            data = file.read().split('\n')
        data = data[17:-1] # skip header lines and ending '' line
        feature = []
        for line in data:
            info = line.split(',')
            info = list(map(float, info))
            feature.append(info[1:])
        feature = np.array(feature)
        if interpolate:
            return self._interpolate_sensor_data(feature)
        return feature
    
    def _interpolate_sensor_data(self, features, samples=TIME_STEPS):
        '''
        Change frequency of sensor data `features` to `samples` number of samples.
        Args: 
            features = A 2-D Numpy array containing sensor values for several time-steps.
            samples  = Number of samples to choose.
        Return:
            Feature values interpolated to `samples` number of samples.
        '''
        
        # resample a 1d col to `samples` number of samples.
        interpolate_1d = lambda col: signal.resample(col, samples)
        
        # Apply interpolation to each column in `features` argument.
        return np.apply_along_axis(interpolate_1d, 0, features)
    
    def _get_class(self, subdir=''):
        '''
        Returns parent directory name, representing the class.
        '''
        return os.path.basename(os.path.normpath(subdir))
    
    def _get_files(self, dir_name='', sort=False):
        files = [os.path.join(dir_name, file) for file in os.listdir(dir_name)]
        if sort:
            assert len(files)//num_sensors<10, "Add functionality for >=10 trails"
            return sorted(files)
        return files
    
    def _get_trails(self, folder=''):
        files = self._get_files(folder, sort=True)
        assert len(files)%num_sensors == 0, "Inconsistent number of sensors in " + folder
        
        return files
    
    def _get_subdirs(self, dir_name=''):
        '''
        Args:
            dir_name = Name of the directory
                       If dir_name == '', the code uses `self.path` as dir_name
        Return:
            Paths to sub directories within `dir_name`.
        '''
        if dir_name == '':
            dir_name = self.path
        subdirs = os.listdir(dir_name)
        subdirs = [os.path.join(dir_name, sd) for sd in subdirs]
        return list(filter(os.path.isdir, subdirs))
    
    def _format_dataset(self):
        print("Formatting dataset...")
        readings = None
        labels = []
        for key in self.dataset:
            class_no = self.classes.index(key)
            if class_no >= 9:
                class_no = 1
            else:
                class_no = 0
            features = self.dataset[key]
            num_samples = features.shape[0]
            labels.extend(num_samples*[class_no])
            try:
                readings = np.concatenate((readings, features), axis=0)
            except ValueError:
                readings = features
        labels = np.array(labels)
        self.dataset = Data(readings, labels)

    def next_batch(self, batch_size=16, training=True, validation=False):
        features = self.dataset.features
        labels = self.dataset.labels

        train_size = int(features.shape[0] * 0.70)
        valid_size = int(features.shape[0] * 0.85)
        if training:
            features = features[:train_size]
            labels = labels[:train_size]

        elif validation:
            features = features[train_size:valid_size]
            labels = labels[train_size:valid_size]

        else:
            features = features[valid_size:]
            labels = labels[valid_size:]

        num_points = features.shape[0]
        n_batches = int(np.ceil(num_points/batch_size))

        for b in tqdm(range(n_batches)):
            start = b*batch_size
            end = start+batch_size
            if end >= num_points:
                continue

            feature_batch = features[start: end]
            label_batch = labels[start: end]

            yield feature_batch, label_batch