import warnings 

__MODELS__ = {} 

def get_model(name, **kwargs): 
    if name not in __MODELS__.keys(): 
        raise KeyError(f'Model {name} not registered!')
    return __MODELS__[name](**kwargs) 

def register_model(name): 
    def wrapper(cls): 
        if __MODELS__.get(name, None): 
            if __MODELS__[name] != cls: 
                warnings.warn(f'Model {name} already registered!', UserWarning) 
        cls.name = name 
        __MODELS__[name] = cls 
        return cls 
    return wrapper 

def get_model_names(): 
    print(list(__MODELS__.keys())) 