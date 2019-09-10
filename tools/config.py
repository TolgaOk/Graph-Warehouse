import numpy as np
import pickle
import yaml
import os


dict(
    env_kwargs={},
    env_class= ?,
    model_structure= ?,
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

    def save(self, path):
        if os.path.exists(path) and not self.overwrite:
            raise FileExistsError("Overwrite is False!")
        elif os.path.exists(path) and self.overwrite:
            pickle.dump(self.info, open(path, "wb"))
            # yaml.dump()
        else:
            os.makedirs(path, exist_ok=True)
            pickle.dump(self.info, open(path, "wb"))
    
    def info_view(self):
        """ Read only informations for serializable objects.
        """
        view_dict = dict(
            env_kwargs=self.info["env_kwargs"],
            env_class=self.info["env_class"],
            model_structure=self.info["model_structure"],
            model_kwargs=self.info["model_kwargs"],
            model_class=self.info["model_class"],
            hyperparams=self.info["hyperparams"],
            schedular_args=self.info["schedular_args"])
        

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Info file not found at {}".format(path))
        else:
            self.info = pickle.load(open(path, "rb"))

    def create_model(self):
        pass

    def create_env(self):
        pass

    @property
    def model_params(self):
        pass
