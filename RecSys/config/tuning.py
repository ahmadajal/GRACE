import os
import yaml
import itertools
from RecSys.utils.config import get_default_config


def generate_tuning(hyperparams, configs_dir="LightGCN/config/tuning/"):
    """Generate tuning configuration files.

    Args:
        hyperparams (dict): {hyperparameters to tune: range of possible values}.
        configs_dir (str): path to directory where to save the configuration files.
    """
    if not os.path.exists(configs_dir):
        os.makedirs(configs_dir)

    config = get_default_config()

    keys, values = zip(*hyperparams.items())
    for v in itertools.product(*values):
        config.update(dict(zip(keys, v)))
        config["name"] = "_".join([f"{k}-{v}" for k, v in zip(keys, v)])
        with open(os.path.join(configs_dir, f"{config['name']}.yaml"), 'w') as f:
            yaml.dump(config, f)

    config["epochs"] = 100
    config["early_stopping"] = "val_Rec@25_ne"

    run_all = f"""for config_name in $(ls {configs_dir})
    do
        echo $config_name
        python main_LightGCN.py {configs_dir}$config_name {os.path.join(configs_dir, "res")}
    done"""
    with open(os.path.join(configs_dir, "run_all.sh"), 'w') as f:
        f.write(run_all)

    print("Number of configurations:", len(list(itertools.product(*values))))
    print(f"Run all configurations with: sh {os.path.join(configs_dir, 'run_all.sh')}")
