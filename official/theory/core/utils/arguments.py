from omegaconf.omegaconf import OmegaConf
import json


class TrainArgs():
    def __init__(self,):
        self.model_cfg = {}
        self.hw_cfg = {}
        self.res_cfg = {}
        self.trained_samples = 0
        self.throughput = -1

    def add(self, key, attr):
        if hasattr(self, key):
            setattr(self, key, attr)
        else:
            print('Err set:', key, attr)

    @staticmethod
    def convert_json_cfg(cfg, hw_cfg):
        model_type = cfg['model_type']
        new_cfg = {
            'hidden_size': cfg['hidden_size'],
            'encoder_seq_length': cfg['max_position_embeddings'],
            'num_attention_heads': cfg['num_attention_heads'],
            'num_layers': cfg['num_hidden_layers'],
            # 'kv_channels': cfg['num_key_value_heads'],
            'precision': cfg['torch_dtype'],
            'vocab_size': cfg['vocab_size']
        }
        return new_cfg, hw_cfg, model_type

    @staticmethod
    def convert_hydra_cfg(cfg, hw_cfg):
        n_device = None
        if 'run' in cfg:
            model_type = cfg['run']['name']
        elif 'name' in cfg:
            model_type = cfg['name']
        else:
            model_type = None
        def convert_dict_value(_str):
            try:
                return eval(_str)
            except:
                return str(_str)
        if 'model' in cfg and isinstance(cfg['model'], dict):
            new_cfg = cfg['model']
        elif 'base' in cfg:
            new_cfg = cfg['base']
        else:
            new_cfg = {k: convert_dict_value(cfg[k]) for k in cfg}
        if not new_cfg.get('precision'):
            if 'trainer' in cfg:
                new_cfg['precision'] = cfg['trainer']['precision']
            elif 'bf16' in cfg and cfg['bf16']:
                new_cfg['precision'] = 'bf16'
            elif 'fp16' in cfg and cfg['fp16']:
                new_cfg['precision'] = 'fp16'
            elif 'fp8' in cfg and cfg['fp8']:
                new_cfg['precision'] = 'fp8'
            elif 'bf16' in new_cfg and new_cfg['bf16']:
                new_cfg['precision'] = 'bf16'
            elif 'fp16' in new_cfg and new_cfg['fp16']:
                new_cfg['precision'] = 'fp16'
            elif 'fp8' in new_cfg and new_cfg['fp8']:
                new_cfg['precision'] = 'fp8'
            else:
                new_cfg['precision'] = 'fp32'
        if not new_cfg.get('max_steps'):
            if 'trainer' in cfg:
                new_cfg['max_steps'] = cfg['trainer']['max_steps']
        if not hw_cfg.get('n_nodes'):
            if 'trainer' in cfg:
                hw_cfg['n_nodes'] = cfg['trainer']['num_nodes']
        elif 'trainer' in cfg and hw_cfg['n_nodes'] != cfg['trainer']['num_nodes']:
            print('Not matched:', hw_cfg['n_nodes'], cfg['trainer']['num_nodes'])
        if 'trainer' in cfg:
            n_device = cfg['trainer']['devices'] * cfg['trainer']['num_nodes']
        if not hw_cfg.get('n_device'):
            hw_cfg['n_device'] = n_device
        elif n_device and hw_cfg['n_device'] != n_device:
            print('Not matched:', hw_cfg['n_device'], n_device)
            hw_cfg['n_device'] = n_device
        if new_cfg.get('nsys_profile') and new_cfg['nsys_profile']['enabled']:
            new_cfg['nsys_steps'] = new_cfg['nsys_profile']['end_step'] - \
                new_cfg['nsys_profile']['start_step']
            if 'run' not in cfg:
                new_cfg['nsys_steps'] += 1
            new_cfg['nsys_ranks'] = new_cfg['nsys_profile']['ranks']
        elif new_cfg.get('profile') and new_cfg['profile']:
            new_cfg['nsys_steps'] = new_cfg['profile_step_end'] - \
                new_cfg['profile_step_start']
            new_cfg['nsys_ranks'] = new_cfg['profile_ranks']
        return new_cfg, hw_cfg, model_type

    def load(self, key):
        file = getattr(self, key)
        if isinstance(file, str):
            if file.endswith('json'):
                with open(file, encoding='utf-8-sig', errors='ignore') as f:
                    setattr(self, key, json.load(f, strict=False))
                return 'json'
            elif file.endswith('yaml') or file.endswith('yml'):
                config = OmegaConf.load(file)
                setattr(self, key, OmegaConf.to_container(config, resolve=True))
                return 'hydra'
            else:
                print('Err:', 'input config not support!')
                exit(0)
        return ''

    def convert_model_cfg(self):
        typ = self.load('model_cfg')
        if typ == 'hydra':
            self.model_cfg, self.hw_cfg, model_type = self.convert_hydra_cfg(self.model_cfg, self.hw_cfg)
        if typ == 'json':
            self.model_cfg, self.hw_cfg, model_type = self.convert_json_cfg(self.model_cfg, self.hw_cfg)

    def clear(self):
        self.model_cfg = {}
        self.hw_cfg = {}
        self.res_cfg = {}
        self.trained_samples = 0
        self.throughput = -1

    @staticmethod
    def convert_res(cfg):
        res = cfg
        metrics = {}
        train_cfg = {}
        if cfg.get('eval_loss'):
            res['eval_loss'] = cfg['eval_loss']
        if cfg.get('eval_runtime'):
            res['validation_epoch_timing'] = cfg['eval_runtime']
        if cfg.get('eval_steps_per_second'):
            res['validation_step_timing'] = 1 / cfg['eval_steps_per_second']
        if cfg.get('train_loss'):
            res['train_loss'] = cfg['train_loss']
        if cfg.get('train_runtime'):
            res['total_time'] = cfg['train_runtime']
        if cfg.get('train_samples_per_second'):
            res['tps_per_device'] = cfg['train_samples_per_second']
        if cfg.get('train_steps_per_second'):
            res['train_step_timing'] = 1 / cfg['train_steps_per_second']
        if cfg.get('reduced_train_loss'):
            res['train_loss'] = cfg['reduced_train_loss']
        if cfg.get('global_step'):
            train_cfg['max_steps'] = cfg['global_step']
        if cfg.get('consumed_samples'):
            train_cfg['trained_samples'] = cfg['consumed_samples']
        if cfg.get('total_flos'):
            metrics['total_tflops'] = cfg['total_flos'] / 1000000000000
        if cfg.get('params'):
            metrics['model_params'] = cfg['params']
        if cfg.get('estimated_params'):
            metrics['estimated_params'] = cfg['estimated_params']
        if cfg.get('gradient_accumulation_steps'):
            if train_cfg.get('gradient_accumulation_steps') != cfg.get('gradient_accumulation_steps'):
                train_cfg['gradient_accumulation_steps'] = cfg.get('gradient_accumulation_steps')
        return res, metrics, train_cfg
