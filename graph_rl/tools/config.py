import numpy as np
import pickle
import yaml
import os
import sys

from graph_rl.models.knowledgenet import GraphA2C
from graph_rl.environments.warehouse import VariationalWarehouse

class Config:

    def __init__(self, info, overwrite=False):
        self.overwrite = overwrite
        
        for key in info.keys():
            setattr(self, key, info[key])
        self.attr_keys = info.keys()

    def save(self, path, suffix=0):
        """
        Save model data and info to the given path. Note that path does 
        not contain the name.
        """
        if os.path.exists(path) and not self.overwrite:
            raise FileExistsError("Overwrite is False!")
        else :
            os.makedirs(path, exist_ok=True)
            os.makedirs(path + "/params", exist_ok=True)
            info = {key: getattr(self, key) for key in self.attr_keys 
                    if key != "model_params"}
            pickle.dump(info, open(path + "/data.b", "wb"))
            pickle.dump(self.model_params, open(path + "/params/param_" + str(suffix) + ".b", "wb"))
            yaml.dump(self.info_view(), open(path + "/info.yaml", "w"))

    def info_view(self):
        """ Read only informations for serializable objects.
        """
        view_dict = dict(
            env_class=self.env_class,
            model_structure=None,
            model_kwargs={key: value for key, value in self.model_kwargs.items() 
                          if isinstance(value,(str,int,float,tuple,list,dict)) 
                          and len(str(value))<100},
            model_class=self.model_class,
            hyperparams=self.hyperparams)
        return view_dict
        
    @staticmethod
    def load(path, overwrite=False, suffix=0):
        """ Load data from the given path. Note that given argument <param> 
        does not contain the name.
        """

        if not os.path.exists(path):
            raise FileNotFoundError("Config file not found at {}".format(path))
        else:
            info = pickle.load(open(path + "/data.b", "rb"))
            params = pickle.load(open(path + "/params/param_" + str(suffix) + ".b", "rb"))
            info["model_params"] = params
        return Config(info, overwrite=overwrite)
        
    def initiate_model(self):
        return self.model_class(**self.model_kwargs)

    def initiate_env(self):
        return self.env_class(**self.env_kwargs)
