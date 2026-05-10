import hydra
import lightning as L

import diffusion_gosai_cfg as diffusion_gosai
import main_gosai as main_gosai_base
import utils


# Reuse the existing training/tokenizer/dataloader pipeline, but swap in the
# classifier-free-guidance diffusion module for this entrypoint.
main_gosai_base.diffusion_gosai = diffusion_gosai


@hydra.main(
  version_base=None,
  config_path='configs_gosai',
  config_name='config_cfg_mrl50')
def main(config):
  """Train the 50nt MRL CFG diffusion model."""
  L.seed_everything(config.seed)
  main_gosai_base._print_config(config, resolve=True, save_cfg=True)
  logger = utils.get_logger(__name__)
  assert config.mode == 'train'
  main_gosai_base._train(config, logger)


if __name__ == '__main__':
  main()
