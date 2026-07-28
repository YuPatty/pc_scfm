'''
Decorator for each dataset
The dataset should be registered by the decorator @register_dataset(name). 
The dataset can be retrieved by get_dataset(name, **kwargs). 
The supported datasets can be retrieved by get_dataset_names().
'''

import warnings 

__DATASETS__ = {} 

def register_dataset(name): 
    def wrapper(cls): 
        # Verify name is not already registered 
        if __DATASETS__.get(name, None): 
            if __DATASETS__[name] != cls: 
                warnings.warn("Dataset already registered!", UserWarning) 
        # register dataset class 
        cls.name = name 
        __DATASETS__[name] = cls 
        return cls 
    return wrapper 

def get_dataset(name, **kwargs):
    if name not in __DATASETS__.keys(): 
        raise NameError(f"Dataset '{name}' is not defined.")
    return __DATASETS__[name](**kwargs) 

def get_dataset_names(): 
    print(list(__DATASETS__.keys()))