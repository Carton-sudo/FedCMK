import json
import os

__all__ = ["ConfLoader", "directory_setter", "config_overwriter"]


class ConfLoader:
    """
    Load json config file using DictWithAttributeAccess object_hook.
    ConfLoader(conf_name).opt attribute is the result of loading json config file.
    """

    class DictWithAttributeAccess(dict):
        """
        This inner class makes dict to be accessed same as class attribute.
        For example, you can use opt.key instead of the opt['key']
        """

        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    def __init__(self, conf_name):
        self.conf_name = conf_name
        self.opt = self.__get_opt()

    def __load_conf(self):
        with open(self.conf_name, "r") as conf:
            opt = json.load(
                conf, object_hook=lambda dict: self.DictWithAttributeAccess(dict)
            )
        return opt

    def __get_opt(self):
        opt = self.__load_conf()
        opt = self.DictWithAttributeAccess(opt)

        return opt


def directory_setter(path="./results", make_dir=False):
    """
    Make dictionary if not exists.
    """
    if not os.path.exists(path) and make_dir:
        os.makedirs(path)  # make dir if not exist
        print("directory %s is created" % path)

    if not os.path.isdir(path):
        raise NotADirectoryError(
            "%s is not valid. set make_dir=True to make dir." % path
        )


def config_overwriter(opt, args):
    """
    Overwrite loaded configuration by parsing arguments.
    """
    if args.dataset_name is not None:
        opt.data_setups.dataset_name = args.dataset_name

    if args.batch_size is not None:
        opt.data_setups.batch_size = args.batch_size

    if args.n_clients is not None:
        opt.data_setups.n_clients = args.n_clients

    if args.partition_method is not None:
        opt.data_setups.partition.method = args.partition_method

    if args.partition_s is not None:
        opt.data_setups.partition.shard_per_user = args.partition_s

    if args.partition_alpha is not None:
        opt.data_setups.partition.alpha = args.partition_alpha

    partition_seed = getattr(args, "partition_seed", None)
    if partition_seed is not None:
        opt.data_setups.partition.seed = partition_seed

    partition_index_path = getattr(args, "partition_index_path", None)
    if partition_index_path is not None:
        opt.data_setups.partition.index_path = partition_index_path

    if args.model_name is not None:
        opt.train_setups.model.name = args.model_name

    if args.n_rounds is not None:
        opt.train_setups.scenario.n_rounds = args.n_rounds

    if args.sample_ratio is not None:
        opt.train_setups.scenario.sample_ratio = args.sample_ratio

    if args.local_epochs is not None:
        opt.train_setups.scenario.local_epochs = args.local_epochs

    if args.device is not None:
        opt.train_setups.scenario.device = args.device

    if args.lr is not None:
        opt.train_setups.optimizer.params.lr = args.lr

    if args.momentum is not None:
        opt.train_setups.optimizer.params.momentum = args.momentum

    if args.wd is not None:
        opt.train_setups.optimizer.params.weight_decay = args.wd

    if args.algo_name is not None:
        opt.train_setups.algo.name = args.algo_name

    if args.seed is not None:
        opt.train_setups.seed = args.seed

    if args.group is not None:
        opt.wandb_setups.group = args.group

    if args.exp_name is not None:
        opt.wandb_setups.name = args.exp_name

    proto_k = getattr(args, "proto_k", None)
    if proto_k is not None:
        opt.train_setups.algo.params.proto_K = proto_k

    proto_lambda = getattr(args, "proto_lambda", None)
    if proto_lambda is not None:
        opt.train_setups.algo.params.proto_lambda = proto_lambda

    fusion_mode = getattr(args, "fusion_mode", None)
    if fusion_mode is not None:
        opt.train_setups.algo.params.fusion_mode = fusion_mode

    diagnostic_enabled = getattr(args, "diagnostic_enabled", None)
    if diagnostic_enabled is not None:
        opt.train_setups.algo.params.diagnostic_enabled = (
            diagnostic_enabled == "true"
        )

    campaign_diagnostic_enabled = getattr(
        args, "campaign_diagnostic_enabled", None
    )
    if campaign_diagnostic_enabled is not None:
        opt.train_setups.algo.params.campaign_diagnostic_enabled = (
            campaign_diagnostic_enabled == "true"
        )

    sample_prediction_logging_enabled = getattr(
        args, "sample_prediction_logging_enabled", None
    )
    if sample_prediction_logging_enabled is not None:
        opt.train_setups.algo.params.sample_prediction_logging_enabled = (
            sample_prediction_logging_enabled == "true"
        )

    sample_prediction_start_round = getattr(
        args, "sample_prediction_start_round", None
    )
    if sample_prediction_start_round is not None:
        opt.train_setups.algo.params.sample_prediction_start_round = (
            sample_prediction_start_round
        )

    proto_arm = getattr(args, "proto_arm", None)
    if proto_arm is not None:
        opt.train_setups.algo.params.proto_arm = proto_arm

    server_eval_interval = getattr(args, "server_eval_interval", None)
    if server_eval_interval is not None:
        opt.train_setups.algo.params.server_eval_interval = (
            server_eval_interval
        )

    logging_setups = opt.train_setups.setdefault("logging", {})
    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        logging_setups["output_dir"] = output_dir

    diagnostic_dir = getattr(args, "diagnostic_dir", None)
    if diagnostic_dir is not None:
        logging_setups["diagnostic_dir"] = diagnostic_dir

    resume_from = getattr(args, "resume_from", None)
    if resume_from is not None:
        logging_setups["resume_from"] = resume_from

    client_schedule_path = getattr(args, "client_schedule_path", None)
    if client_schedule_path is not None:
        logging_setups["client_schedule_path"] = client_schedule_path

    checkpoint_interval = getattr(args, "checkpoint_interval", None)
    if checkpoint_interval is not None:
        logging_setups["checkpoint_interval"] = checkpoint_interval

    save_model_artifacts = getattr(args, "save_model_artifacts", None)
    if save_model_artifacts is not None:
        logging_setups["save_model_artifacts"] = (
            save_model_artifacts == "true"
        )

    overwrite_output = getattr(args, "overwrite_output", None)
    if overwrite_output is not None:
        logging_setups["overwrite_output"] = overwrite_output == "true"

    return opt
