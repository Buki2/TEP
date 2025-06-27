import os
from omegaconf import OmegaConf

# The default
config_names = os.getenv('CONFIG_NAMES', None)
if config_names is None:
    config_names = 'my_config'  # Modify this if you want to use another default config

print(config_names)
configs = [OmegaConf.load('model/glip_mebow/configs/base_config.yaml')]

if config_names is not None:
    for config_name in config_names.split(','):
        configs.append(OmegaConf.load(f'model/glip_mebow/configs/{config_name.strip()}.yaml'))

# unsafe_merge makes the individual configs unusable, but it is faster
config = OmegaConf.unsafe_merge(*configs)

print(config)