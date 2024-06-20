import os



def get_absolute_file_path(file_path):
    if file_path.startswith("/"):
        return file_path
    else:
        return file_path
    
def merge_config(config, base_config):
    for key, _ in config.items():
        if isinstance(config[key], dict) and key not in base_config:
            base_config[key] = config[key]
        elif isinstance(config[key], dict):
            merge_config(config[key], base_config[key])
        else:
            if key in base_config:
                base_config[key] = config[key]
            else:
                base_config.update({key: config[key]})
    
def read_config(config_file):
    import anyconfig
    if os.path.exists(config_file):
        with open(config_file, "rb") as fr:
            config = anyconfig.load(fr)
        if 'base' in config:
            base_config_path = config['base']
            if not base_config_path.startswith('/'):
                base_config_path = base_config_path
        elif os.path.basename(config_file) == 'base.yaml':
            return config
        else:
            base_config_path = os.path.join(os.path.dirname(config_file), "base.yaml")
        base_config = read_config(base_config_path)
        merged_config = base_config.copy()
        merge_config(config, merged_config)
        return merged_config
    else:
        return {}
    

def save_params(save_dir, save_json, yml_name='config.yaml'):
    import yaml
    with open(os.path.join(save_dir, yml_name), 'w', encoding='utf-8') as f:
        yaml.dump(save_json, f, default_flow_style=False, encoding='utf-8', allow_unicode=True)

def load_config(config_file, experiment_name):
    if not config_file.startswith("/"):
        config_file = get_absolute_file_path(config_file)
    input_config = read_config(config_file)
    experiment_base_config = read_config(os.path.join('config', experiment_name.lower(), 'base.yaml'))
    merged_config = experiment_base_config.copy()
    merge_config(input_config, merged_config)

    base_config = read_config(os.path.join('config', 'base.yaml'))
    final_merged_config = base_config.copy()
    merge_config(merged_config, final_merged_config)
    return final_merged_config