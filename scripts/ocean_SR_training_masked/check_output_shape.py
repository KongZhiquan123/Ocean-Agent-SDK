#!/usr/bin/env python3
"""
check_output_shape.py

Preflight check for model output shape vs HR target.
"""
import argparse
import inspect
import json
import sys

import yaml
import torch


def load_config(path):
    with open(path, 'r') as stream:
        return yaml.load(stream, yaml.FullLoader)


def emit(result):
    print(f"__shape_check__{json.dumps(result)}__shape_check__", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Check model output shape before training')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--device', default='0', help='CUDA device id or "cpu"')
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get('model', {})
    data_cfg = cfg.get('data', {})
    model_name = model_cfg.get('name')

    if not model_name:
        result = {
            'status': 'error',
            'error': 'Missing model name in config',
            'reason': 'model.name is required',
        }
        emit(result)
        return 0

    try:
        from models import _model_dict, _ddpm_dict
        from datasets import _dataset_dict
    except Exception as exc:
        result = {
            'status': 'error',
            'error': f'Import error: {exc}',
            'reason': 'Failed to import models/datasets registry',
        }
        emit(result)
        return 1

    model_entry = _model_dict.get(model_name)
    diffusion_entry = None
    resshift_names = {'Resshift', 'ResShift'}
    if isinstance(model_entry, dict):
        diffusion_entry = model_entry
    elif model_entry is None:
        diffusion_entry = _ddpm_dict.get(model_name)

    if model_entry is None and diffusion_entry is None and model_name in resshift_names:
        diffusion_entry = {}

    if model_entry is None and diffusion_entry is None:
        result = {
            'status': 'error',
            'error': f'Unknown model: {model_name}',
            'reason': 'Model not found in registry',
        }
        emit(result)
        return 0

    dataset_name = data_cfg.get('name')
    dataset_cls = _dataset_dict.get(dataset_name)
    if dataset_cls is None:
        result = {
            'status': 'error',
            'error': f'Unknown dataset: {dataset_name}',
            'reason': 'Dataset not found in registry',
        }
        emit(result)
        return 0

    device_arg = str(args.device)
    if device_arg.lower() == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device_id = int(device_arg)
        torch.cuda.set_device(device_id)
        device = torch.device(f'cuda:{device_id}')

    try:
        dataset = dataset_cls(data_cfg)
        if len(dataset.valid_dataset) > 0:
            sample = dataset.valid_dataset[0]
        else:
            sample = dataset.train_dataset[0]

        if len(sample) == 3:
            x, y, _ = sample
        else:
            x, y = sample

        if x.dim() == 3:
            x = x.unsqueeze(0)
        if y.dim() == 3:
            y = y.unsqueeze(0)

        if diffusion_entry is not None:
            if x.dim() != 4 or y.dim() != 4:
                result = {
                    'status': 'error',
                    'error': 'Invalid input shapes',
                    'reason': 'Expected inputs in [B,H,W,C] format',
                    'details': {
                        'lr_shape': list(x.shape),
                        'hr_shape': list(y.shape),
                        'model': model_name,
                    },
                }
                emit(result)
                return 0

            lr_h, lr_w, lr_c = x.shape[1], x.shape[2], x.shape[3]
            hr_h, hr_w, hr_c = y.shape[1], y.shape[2], y.shape[3]
            if lr_c != hr_c:
                result = {
                    'status': 'error',
                    'error': 'LR/HR channel mismatch',
                    'reason': 'Input and target channels do not match',
                    'details': {
                        'lr_channels': lr_c,
                        'hr_channels': hr_c,
                        'model': model_name,
                    },
                }
                emit(result)
                return 0

            if hr_h % lr_h != 0 or hr_w % lr_w != 0:
                result = {
                    'status': 'error',
                    'error': 'LR/HR spatial mismatch',
                    'reason': 'HR spatial size is not divisible by LR size',
                    'details': {
                        'lr_shape': [lr_h, lr_w],
                        'hr_shape': [hr_h, hr_w],
                        'model': model_name,
                    },
                }
                emit(result)
                return 0

            inferred_scale_h = hr_h // lr_h
            inferred_scale_w = hr_w // lr_w
            cfg_scale = data_cfg.get('sample_factor', None) or model_cfg.get('scale', None)
            if cfg_scale is not None and (cfg_scale != inferred_scale_h or cfg_scale != inferred_scale_w):
                result = {
                    'status': 'error',
                    'error': 'Scale mismatch',
                    'reason': 'Configured scale does not match LR/HR shapes',
                    'details': {
                        'cfg_scale': cfg_scale,
                        'lr_shape': [lr_h, lr_w],
                        'hr_shape': [hr_h, hr_w],
                        'inferred_scale': [inferred_scale_h, inferred_scale_w],
                        'model': model_name,
                    },
                }
                emit(result)
                return 0

            if 'out_channel' in model_cfg and model_cfg.get('out_channel') != hr_c:
                result = {
                    'status': 'error',
                    'error': 'Model output channels mismatch',
                    'reason': 'out_channel does not match HR channels',
                    'details': {
                        'out_channel': model_cfg.get('out_channel'),
                        'hr_channels': hr_c,
                        'model': model_name,
                    },
                }
                emit(result)
                return 0

            x_nchw = x.permute(0, 3, 1, 2).to(device)
            y_nchw = y.permute(0, 3, 1, 2).to(device)

            if model_name in resshift_names:
                resshift_cfg = cfg.get('resshift')
                if not isinstance(resshift_cfg, dict):
                    result = {
                        'status': 'error',
                        'error': 'Missing resshift config',
                        'reason': 'resshift section is required for Resshift preflight',
                        'details': {
                            'model': model_name,
                        },
                    }
                    emit(result)
                    return 0

                try:
                    from utils.metrics import get_obj_from_str
                    import torch.nn.functional as F

                    model_target = resshift_cfg.get('model', {}).get('target')
                    model_params = resshift_cfg.get('model', {}).get('params', {})
                    diffusion_target = resshift_cfg.get('diffusion', {}).get('target')
                    diffusion_params = resshift_cfg.get('diffusion', {}).get('params', {})

                    if not model_target or not diffusion_target:
                        result = {
                            'status': 'error',
                            'error': 'Invalid resshift config',
                            'reason': 'resshift.model.target and resshift.diffusion.target are required',
                            'details': {
                                'model': model_name,
                            },
                        }
                        emit(result)
                        return 0

                    model = get_obj_from_str(model_target)(**model_params)
                    model = model.to(device)
                    model.eval()

                    base_diffusion = get_obj_from_str(diffusion_target)(**diffusion_params)
                    x_up = F.interpolate(x_nchw, size=y_nchw.shape[2:], mode='bicubic', align_corners=False)
                    tt = torch.randint(0, base_diffusion.num_timesteps, size=(y_nchw.shape[0],), device=device)
                    model_kwargs = {'lq': x_up}
                    with torch.no_grad():
                        losses = base_diffusion.training_losses(
                            model, y_nchw, x_up, tt,
                            first_stage_model=None,
                            model_kwargs=model_kwargs,
                            noise=None,
                        )
                    if isinstance(losses, tuple):
                        losses = losses[0]
                    if not isinstance(losses, dict):
                        result = {
                            'status': 'error',
                            'error': 'ResShift preflight returned invalid losses',
                            'reason': 'training_losses output type is unexpected',
                            'details': {
                                'output_type': str(type(losses)),
                                'model': model_name,
                            },
                        }
                        emit(result)
                        return 0

                    result = {
                        'status': 'ok',
                        'details': {
                            'lr_shape': [lr_h, lr_w, lr_c],
                            'hr_shape': [hr_h, hr_w, hr_c],
                            'model': model_name,
                            'kind': 'diffusion',
                            'note': 'resshift_forward_checked',
                        },
                    }
                    emit(result)
                    return 0
                except Exception as exc:
                    result = {
                        'status': 'error',
                        'error': f'ResShift preflight failed: {exc}',
                        'reason': 'Exception during Resshift preflight',
                        'details': {
                            'model': model_name,
                        },
                    }
                    emit(result)
                    return 1

            model_cls = diffusion_entry.get('model')
            diffusion_cls = diffusion_entry.get('diffusion')
            if model_cls is None or diffusion_cls is None:
                result = {
                    'status': 'ok',
                    'details': {
                        'lr_shape': [lr_h, lr_w, lr_c],
                        'hr_shape': [hr_h, hr_w, hr_c],
                        'model': model_name,
                        'kind': 'diffusion',
                        'note': 'forward_check_skipped',
                    },
                }
                emit(result)
                return 0

            try:
                model = model_cls(model_cfg)
                model = model.to(device)
                model.eval()

                beta_schedule = None
                beta_cfg = cfg.get('beta_schedule', {})
                if isinstance(beta_cfg, dict):
                    beta_schedule = beta_cfg.get('train')

                sig = inspect.signature(diffusion_cls)
                kwargs = {'model_args': model_cfg}
                if 'schedule_opt' in sig.parameters and beta_schedule is not None:
                    kwargs['schedule_opt'] = beta_schedule

                diffusion = diffusion_cls(model, **kwargs)
                if hasattr(diffusion, 'to'):
                    diffusion = diffusion.to(device)
                if hasattr(diffusion, 'eval'):
                    diffusion.eval()

                if hasattr(diffusion, 'set_new_noise_schedule'):
                    if beta_schedule is None:
                        result = {
                            'status': 'error',
                            'error': 'Missing beta_schedule',
                            'reason': 'Diffusion model requires beta_schedule for preflight',
                            'details': {
                                'model': model_name,
                            },
                        }
                        emit(result)
                        return 0
                    diffusion.set_new_noise_schedule(beta_schedule, device=device)
                if hasattr(diffusion, 'set_loss'):
                    diffusion.set_loss(device)

                if callable(diffusion):
                    with torch.no_grad():
                        loss = diffusion({'SR': x_nchw, 'HR': y_nchw})
                    if not isinstance(loss, torch.Tensor):
                        result = {
                            'status': 'error',
                            'error': 'Diffusion output is not a Tensor',
                            'reason': 'Unsupported diffusion output type',
                            'details': {
                                'output_type': str(type(loss)),
                                'model': model_name,
                            },
                        }
                        emit(result)
                        return 0

                result = {
                    'status': 'ok',
                    'details': {
                        'lr_shape': [lr_h, lr_w, lr_c],
                        'hr_shape': [hr_h, hr_w, hr_c],
                        'model': model_name,
                        'kind': 'diffusion',
                    },
                }
                emit(result)
                return 0
            except Exception as exc:
                result = {
                    'status': 'error',
                    'error': f'Diffusion preflight failed: {exc}',
                    'reason': 'Exception during diffusion preflight',
                    'details': {
                        'model': model_name,
                    },
                }
                emit(result)
                return 1

        model = model_entry(model_cfg)
        model = model.to(device)
        model.eval()

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = model(x)

        if not isinstance(y_pred, torch.Tensor):
            result = {
                'status': 'error',
                'error': 'Model output is not a Tensor',
                'reason': 'Unsupported model output type',
                'details': {
                    'output_type': str(type(y_pred)),
                },
            }
            emit(result)
            return 0

        if list(y_pred.shape) != list(y.shape):
            result = {
                'status': 'error',
                'error': 'Model output shape mismatch',
                'reason': 'Output shape does not match HR target',
                'details': {
                    'pred_shape': list(y_pred.shape),
                    'target_shape': list(y.shape),
                    'model': model_name,
                },
            }
            emit(result)
            return 0

        result = {
            'status': 'ok',
            'details': {
                'pred_shape': list(y_pred.shape),
                'target_shape': list(y.shape),
                'model': model_name,
            },
        }
        emit(result)
        return 0
    except Exception as exc:
        result = {
            'status': 'error',
            'error': f'Preflight check failed: {exc}',
            'reason': 'Exception during shape check',
        }
        emit(result)
        return 1


if __name__ == '__main__':
    sys.exit(main())
