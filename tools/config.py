import numpy as np
import pickle
import yaml
import os


dict(
    env_kwargs={},
    env_class= ?,
    model_kwargs= ?,
    model_params= ?,
    model_class= ??,
    hyperparams= ?
    visdom_args= ?,
    logger_config=?,
    schedular_args=?,
)


class Config:

    def __init__(info, overwrite=False):
        self.info = info
        self.overwrite = overwrite
        
        for key in self.info.keys():
            setattr(self.info[key], key)

    def save(self, path):
        """
        Save model data and info to the given path. Note that path does 
        not contain the name.
        """
        if os.path.exists(path) and not self.overwrite:
            raise FileExistsError("Overwrite is False!")
        elif os.path.exists(path) and self.overwrite:
            pickle.dump(self.info, open(path + "data.b", "wb"))
            yaml.dump(self.info_view(), open(path + "info.yaml", "wb"))
        else:
            os.makedirs(path, exist_ok=True)
            pickle.dump(self.info, open(path, "wb"))
    
    def info_view(self):
        """ Read only informations for serializable objects.
        """
        view_dict = dict(
            env_kwargs=self.env_kwargs,
            env_class=self.env_class,
            model_structure=self.model_structure,
            model_kwargs=self.model_kwargs,
            model_class=self.model_class,
            hyperparams=self.hyperparams,
            schedular_args=self.schedular_args)
        return view_dict
        
    def load(self, path):
        """ Load data from the given path. NOte that given argument <param> 
        does not contain the name.
        """
        if not os.path.exists(path):
            raise FileNotFoundError("Info file not found at {}".format(path))
        else:
            self.info = pickle.load(open(path, "rb"))

    def create_model(self):
        return self.model_class(self.model_kwargs)

    def create_env(self):
        return self.env_class(self.env_kwargs)
