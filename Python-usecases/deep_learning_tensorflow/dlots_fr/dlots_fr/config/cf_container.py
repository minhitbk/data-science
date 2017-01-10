""" 
Written by Minh Tran <minhitbk@gmail.com>, Jan 2017.
"""
import json

from cf_parser import config_parser


class Config(object):
    """
    This is a container for configured parameters.
    """
    # Load general parameters
    data_path = config_parser.get("general_param", "data_path")
    num_feature = int(config_parser.get("general_param", "num_feature"))
    num_response = int(config_parser.get("general_param", "num_response"))
    num_epoch = int(config_parser.get("general_param", "num_epoch"))
    batch_size = int(config_parser.get("general_param", "batch_size"))
    max_grad_norm = float(config_parser.get("general_param", "max_grad_norm"))
    learning_rate = float(config_parser.get("general_param", "learning_rate"))
    model_type = config_parser.get("general_param", "model_type")
    save_model = config_parser.get("general_param", "save_model")

    # Load cell parameters
    cell_type = config_parser.get("cell_param", "cell_type")
    cell_size = int(config_parser.get("cell_param", "cell_size"))
    num_cell = int(config_parser.get("cell_param", "num_cell"))

    # Load embedding parameters
    embed = config_parser.get("embedding_param", "embed")
    schema = json.loads(config_parser.get("embedding_param", "schema"))
    embed_size = json.loads(config_parser.get("embedding_param", "embed_size"))

    # Load encoding parameters
    encode = config_parser.get("encoding_param", "encode")

    # Load attention parameters
    attention = config_parser.get("attention_param", "attention")
    att_size = int(config_parser.get("attention_param", "att_size"))
    context_size = int(config_parser.get("attention_param", "context_size"))

    # Load prediction parameters
    layers = json.loads(config_parser.get("prediction_param", "layers"))
    act_func = config_parser.get("prediction_param", "act_func")
    keep_drop = json.loads(config_parser.get("prediction_param", "keep_drop"))
