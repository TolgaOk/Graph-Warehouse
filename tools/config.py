import numpy as np
import pickle
import yaml
import os
import sys

from models.knowledgenet import GraphA2C


class Config:

    def __init__(self, info, overwrite=False):
        self.info = info
        self.overwrite = overwrite
        
        for key in self.info.keys():
            setattr(self, key, self.info[key])

    def save(self, path):
        """
        Save model data and info to the given path. Note that path does 
        not contain the name.
        """
        if os.path.exists(path) and not self.overwrite:
            raise FileExistsError("Overwrite is False!")
        else :
            os.makedirs(path, exist_ok=True)
            pickle.dump(self.info, open(path + "/data.b", "wb"))
            yaml.dump(self.info_view(), open(path + "/info.yaml", "w"))
    def info_view(self):
        """ Read only informations for serializable objects.
        """

        view_dict = dict(
            env_class=self.env_class,
            model_structure=None,
            model_kwargs={key: value for key, value in self.model_kwargs.items() 
                          if isinstance(value,(str,int,float,tuple,list,dict)) 
                          and sys.getsizeof(value)<100},
            model_class=self.model_class,
            hyperparams=self.hyperparams)
        return view_dict
        
    @staticmethod
    def load(path):
        """ Load data from the given path. NOte that given argument <param> 
        does not contain the name.
        """
        if not os.path.exists(path):
            raise FileNotFoundError("Info file not found at {}".format(path))
        else:
            info = pickle.load(open(path + "/data.b", "rb"))
        return info
        
    def initiate_model(self):
        return self.model_class(self.model_kwargs)

    def initiate_env(self):
        return self.env_class(self.env_kwargs)
