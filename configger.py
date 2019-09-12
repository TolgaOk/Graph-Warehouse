import yaml
import argparse

from tools.config import Config
from environments.warehouse import VariationalWarehouse
from models.knowledgenet import GraphA2C
from models.ourattnnet import OurAttnNet
from models.relationalnet import RelationalNet
from models.vanillanet import ConvModel


def generate_config(save_name, env_config, model_config, hyperparameter_config,
                    logger_config, forced):
    path = "configs"
    environment = yaml.load(
        open(path + "/env_configs/" + env_config))["env_class"]
    network = yaml.load(
        open(path + "/model_configs/" + model_config))["model_class"]
    environment_kwargs = environment.prep_env(
        path + "/env_configs/" + env_config)
    model_kwargs = yaml.load(
        open(path + "/model_configs/" + model_config, "r"))
    assert sum(key in model_kwargs.keys()
               for key in environment_kwargs.keys()) == 0, (
        "There is a conflict between model"
        " arguments and environment arguments")
    model_kwargs = {**environment_kwargs, **model_kwargs}

    hyperparams = yaml.load(
        open(path + "/hyperparam_configs/" + hyperparameter_config, "r"))
    logger_config = yaml.load(
        open(path + "/logger_configs/" + logger_config, "r"))

    config = Config(dict(env_kwargs=environment_kwargs,
                         env_class=environment,
                         model_kwargs=model_kwargs,
                         model_params=None,
                         model_class=network,
                         hyperparams=hyperparams,
                         logger_config=logger_config), overwrite=forced)
    config.save(path + "/configs/" + save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("--name", help="Config file name to save",
                        action="store", dest="save_name")
    parser.add_argument("--env", help="Environment config file name",
                        action="store", dest="env_config",
                        default="warehouse_3.yaml")
    parser.add_argument("--model", help="Model config file name",
                        action="store", dest="model_config",
                        default="knowledgenet.yaml")
    parser.add_argument("--hyperparameter",
                        help="Hyperparameter config file name",
                        action="store", dest="hyperparameter_config",
                        default="a2c_default.yaml")
    parser.add_argument("--logger", help="Logger config file name",
                        action="store", dest="logger_config",
                        default="default_logger.yaml")
    parser.add_argument(
        "--forced", help="force to overwrite", action='store_true')

    kwargs = parser.parse_args()
    generate_config(**vars(kwargs))
