import os
from pathlib import Path
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import dataloader_gosai
import diffusion_gosai_update as diffusion_gosai
import utils
import random
import string
import datetime
import wandb
import json
omegaconf.OmegaConf.register_new_resolver("uuid", lambda: ''.join(random.choice(string.ascii_letters) for _ in range(10))+'_'+str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")), use_cache=False)
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config):
  if 'hf' in config.backbone:
    return diffusion_gosai.Diffusion(
      config, 
      ).to('cuda')
  
  return diffusion_gosai.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    os.makedirs(config.checkpointing.save_dir, exist_ok=True)
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, test_ds):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds), ('test', test_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch seqs.shape', batch['seqs'].shape)
    print(f'tokens:', dataloader_gosai.dna_detokenize(batch['seqs'][0]))
    print('ids:', batch['seqs'][0])
    

    

def _train(config, logger):
  logger.info('Starting Training.')
  wandb_logger = None
  wandb_settings = wandb.Settings(
      base_url='https://api.wandb.ai'  # Specify your wandb host URL here
  )
  if config.get('wandb', None) is not None and not config.debug_mode:
    if omegaconf.OmegaConf.is_config(config.wandb):
      omegaconf.OmegaConf.set_struct(config.wandb, False)
    config.wandb.mode = 'online'
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      settings=wandb_settings,
      **config.wandb)

  resume_from_ckpt = config.checkpointing.get('resume_from_ckpt', False)
  resume_ckpt_path = config.checkpointing.get('resume_ckpt_path', None)
  if (resume_from_ckpt
      and resume_ckpt_path is not None
      and utils.fsspec_exists(resume_ckpt_path)):
    ckpt_path = resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  tokenizer_type = str(config.data.get('tokenizer_type', 'csv_motif')).lower()

  if tokenizer_type in ('simple_vocab', 'simple'):
    vocab_json = config.data.get('tokenizer_vocab_path')
    if vocab_json is None:
      raise ValueError('tokenizer_type="simple_vocab" expects data.tokenizer_vocab_path in the config.')
    vocab_json = Path(vocab_json)
    if not vocab_json.exists():
      raise FileNotFoundError(f"Vocabulary JSON not found at {vocab_json}")
    with vocab_json.open() as fp:
      vocab_dict = json.load(fp)
    tokenizer = dataloader_gosai.SimpleVocabTokenizer(
      vocab_dict,
      pad_token=config.data.get('pad_token', 'N'),
      eos_token=config.data.get('eos_token', 'EOS'),
      unk_token=config.data.get('unk_token', None),
      normalize_case=config.data.get('normalize_case', True),
    )
    pad_id = tokenizer.pad_token_id

  elif tokenizer_type == 'csv_motif':
    vocab_json = config.data.get('motif_vocab_path')
    if vocab_json is None:
      raise ValueError('tokenizer_type="csv_motif" now expects data.motif_vocab_path (JSON) in the config.')
    vocab_json = Path(vocab_json)
    if not vocab_json.exists():
      raise FileNotFoundError(f"Motif vocabulary JSON not found at {vocab_json}")
    tokenizer = dataloader_gosai.MotifAwareTokenizer(
      vocab_json_path=vocab_json,
      pad_token=config.data.get('pad_token', 'N'),
      eos_token=config.data.get('eos_token', 'EOS'),
      base_tokens=config.data.get('motif_base_tokens', ("A", "C", "G", "T")),
      max_length=config.model.length,
      trim_to=config.data.get('motif_trim_len'))
    pad_id = tokenizer.pad_token_id
  else:
    vocab_path = config.data.get('motif_vocab_path')
    if vocab_path is None:
      raise ValueError('motif tokenizer requires data.motif_vocab_path to be set.')
    vocab_path = Path(vocab_path)
    if not vocab_path.exists():
      raise FileNotFoundError(f"Motif vocabulary not found at {vocab_path}")
    use_fimo = bool(config.data.get('use_fimo', True))
    fimo_path = None
    if use_fimo and config.data.get('fimo_tsv_path'):
      fimo_path = Path(config.data.fimo_tsv_path)
      if not fimo_path.exists():
        raise FileNotFoundError(f"FIMO results not found at {fimo_path}")

    tokenizer = dataloader_gosai.MotifTokenizer(
      vocab_path=vocab_path,
      fimo_path=fimo_path,
      pad_token=config.data.get('pad_token', '<pad>'),
      max_length=config.model.length,
    )
    pad_id = tokenizer.pad_token_id


  # tokenizer,pad_id = dataloader_gosai.build_simple_tokenizer(
  # config.data.tokenizer_vocab_path,
  # )


  # # train_ds, valid_ds, test_ds = dataloader_gosai.get_dataloaders_gosai(config)
  # train_ds, valid_ds, test_ds,pad_id = dataloader_gosai.get_dataloaders_utr(
  #   config,
  #   tokenizer,
  #   pad_id,
  #   csv_path=config.data.utr_csv_path,
  #   max_length=config.model.length,
  #   skip_valid=config.get('skip_validation', False),
  # )


  train_ds, valid_ds, test_ds = dataloader_gosai.get_dataloaders_utr(
    config,
    tokenizer,
    pad_id,
    csv_path=config.data.get('train_csv_path', config.data.get('utr_csv_path')),
    valid_csv_path=config.data.get('valid_csv_path'),
    test_csv_path=config.data.get('test_csv_path'),
    # fasta_path=config.data.get('utr_fasta_path'),
    seq_col=config.data.get('seq_column', 'seq'),
    label_col=config.data.get('label_column', 'auto'),
    max_length=config.model.length,
    skip_valid=config.get('skip_validation', False),
  )
  # freqs = dataloader_gosai.count_token_frequencies(
  #   dataloader=train_ds,  # the training DataLoader
  #   tokenizer=tokenizer,   # your MotifAwareTokenizer instance
  #   output_path="token_frequencies.json",
  #   include_pad=False,    # set True to count pad tokens
  # )
  # print("Saved counts to token_frequencies.json")



  if omegaconf.OmegaConf.is_config(config.data):
    omegaconf.OmegaConf.set_struct(config.data, False)
  config.data.pad_token_id = pad_id
  config.data.vocab_size = tokenizer.get_vocab_size()

  model = diffusion_gosai.Diffusion(
    config, 
    eval=False,
    )

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  print('Start training...')
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs_gosai',
            config_name='config_gosai_pretrain')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  logger = utils.get_logger(__name__)
  assert config.mode == 'train'
  _train(config, logger)


if __name__ == '__main__':
  main()
