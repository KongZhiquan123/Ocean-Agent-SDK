#!/usr/bin/env python3
"""
@file prepare_workspace.py
@description Prepare training workspace: copy only model-relevant code files.
             Adapted for ocean forecast training (no diffusion models).
@author Leizheng
@date 2026-02-26
@version 1.0.0

@changelog
  - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training

Usage:
    python prepare_workspace.py \
        --source_dir /path/to/ocean-forecast-training \
        --target_dir /path/to/workspace \
        --model_name FNO2d \
        --data_name ocean_forecast_npy
"""
import argparse
import inspect
import json
import os
import re
import shutil
import sys


def get_package_name(cls):
    """Get the package directory name for a class."""
    src_file = inspect.getfile(cls)
    return os.path.basename(os.path.dirname(src_file))


def strip_header_docstring(content):
    """Remove module-level docstring (keep shebang)."""
    pattern = r'^((?:#!.*\n)?)\s*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')\s*\n?'
    return re.sub(pattern, r'\1', content, count=1)


def _is_up_to_date(src, dst):
    if not os.path.exists(dst):
        return False
    return os.path.getmtime(dst) >= os.path.getmtime(src)


def copy_file(src, dst):
    if _is_up_to_date(src, dst):
        return False
    if src.endswith('.py'):
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        content = strip_header_docstring(content)
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(content)
        shutil.copystat(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def write_if_changed(path, content):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            if f.read() == content:
                return False
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return True


def copy_dir(src, dst):
    for dirpath, dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        dst_dir = os.path.join(dst, rel)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in filenames:
            copy_file(os.path.join(dirpath, fname),
                      os.path.join(dst_dir, fname))


def main():
    parser = argparse.ArgumentParser(description='Prepare forecast training workspace')
    parser.add_argument('--source_dir', required=True, help='Training framework source directory')
    parser.add_argument('--target_dir', required=True, help='Workspace target directory')
    parser.add_argument('--model_name', required=True, help='Model name')
    parser.add_argument('--data_name', default='ocean_forecast_npy', help='Dataset name')
    args = parser.parse_args()

    src = os.path.abspath(args.source_dir)
    dst = os.path.abspath(args.target_dir)
    model_name = args.model_name
    data_name = args.data_name

    # Add source dir to path for importing registries
    sys.path.insert(0, src)

    from models import MODEL_REGISTRY
    from trainers import TRAINER_REGISTRY
    from datasets import DATASET_REGISTRY

    if model_name not in MODEL_REGISTRY:
        print(json.dumps({
            'status': 'error',
            'error': f'Unknown model: {model_name}',
            'available': list(MODEL_REGISTRY.keys()),
        }))
        sys.exit(1)

    if model_name not in TRAINER_REGISTRY:
        print(json.dumps({
            'status': 'error',
            'error': f'Model {model_name} has no registered trainer',
        }))
        sys.exit(1)

    copied = []

    # ================================================================
    # 1. Core files (entry scripts, utility scripts)
    # ================================================================
    os.makedirs(dst, exist_ok=True)

    core_files = [
        'main.py', 'main_ddp.py', 'config.py',
        'generate_config.py', 'generate_training_report.py',
        'generate_training_plots.py', 'generate_predict_plots.py',
        'validate_dataset.py', 'check_gpu.py', 'list_models.py',
        'recommend_hyperparams.py',
    ]
    for f in core_files:
        s = os.path.join(src, f)
        if os.path.exists(s):
            copy_file(s, os.path.join(dst, f))
            copied.append(f)

    # ================================================================
    # 2. utils/ (complete copy, small and universal)
    # ================================================================
    for d in ['utils']:
        s = os.path.join(src, d)
        t = os.path.join(dst, d)
        if os.path.isdir(s):
            copy_dir(s, t)
            copied.append(f'{d}/')

    # ================================================================
    # 3. models/: only copy the selected model's package
    # ================================================================
    model_cls = MODEL_REGISTRY[model_name]
    cls_name = model_cls.__name__
    pkg_name = get_package_name(model_cls)

    models_dst = os.path.join(dst, 'models')
    os.makedirs(models_dst, exist_ok=True)

    copy_dir(os.path.join(src, 'models', pkg_name),
             os.path.join(models_dst, pkg_name))
    copied.append(f'models/{pkg_name}/')

    init_code = (
        f'from .{pkg_name} import {cls_name}\n\n'
        f'MODEL_REGISTRY = {{\n'
        f'    "{model_name}": {cls_name},\n'
        f'}}\n'
    )
    write_if_changed(os.path.join(models_dst, '__init__.py'), init_code)
    copied.append('models/__init__.py (generated)')

    # ================================================================
    # 4. trainers/: base.py (all forecast models use BaseTrainer)
    # ================================================================
    trainer_cls = TRAINER_REGISTRY[model_name]
    trainer_cls_name = trainer_cls.__name__

    trainers_dst = os.path.join(dst, 'trainers')
    os.makedirs(trainers_dst, exist_ok=True)

    copy_file(os.path.join(src, 'trainers', 'base.py'),
              os.path.join(trainers_dst, 'base.py'))
    copied.append('trainers/base.py')

    trainer_init = (
        'from .base import BaseTrainer\n\n'
        'TRAINER_REGISTRY = {\n'
        f'    "{model_name}": {trainer_cls_name},\n'
        '}\n'
    )
    write_if_changed(os.path.join(trainers_dst, '__init__.py'), trainer_init)
    copied.append('trainers/__init__.py (generated)')

    # ================================================================
    # 5. forecastors/: base.py (forecast inference helpers)
    # ================================================================
    forecastors_src = os.path.join(src, 'forecastors')
    if os.path.isdir(forecastors_src):
        forecastors_dst = os.path.join(dst, 'forecastors')
        os.makedirs(forecastors_dst, exist_ok=True)

        copy_file(os.path.join(forecastors_src, 'base.py'),
                  os.path.join(forecastors_dst, 'base.py'))
        copied.append('forecastors/base.py')

        # Copy __init__.py as-is
        init_src = os.path.join(forecastors_src, '__init__.py')
        if os.path.exists(init_src):
            copy_file(init_src, os.path.join(forecastors_dst, '__init__.py'))
            copied.append('forecastors/__init__.py')

    # ================================================================
    # 6. datasets/: only copy the selected dataset
    # ================================================================
    datasets_dst = os.path.join(dst, 'datasets')
    os.makedirs(datasets_dst, exist_ok=True)

    # Always copy base.py if it exists
    base_ds = os.path.join(src, 'datasets', 'base.py')
    if os.path.exists(base_ds):
        copy_file(base_ds, os.path.join(datasets_dst, 'base.py'))
        copied.append('datasets/base.py')

    if data_name in DATASET_REGISTRY:
        ds_cls = DATASET_REGISTRY[data_name]
        ds_file = inspect.getfile(ds_cls)
        ds_basename = os.path.basename(ds_file)
        ds_cls_name = ds_cls.__name__
        ds_module = ds_basename.replace('.py', '')

        copy_file(ds_file, os.path.join(datasets_dst, ds_basename))
        copied.append(f'datasets/{ds_basename}')

        ds_init = (
            'from .base import BaseDataset\n'
            f'from .{ds_module} import {ds_cls_name}\n\n'
            f'DATASET_REGISTRY = {{\n'
            f'    "{data_name}": {ds_cls_name},\n'
            f'}}\n'
        )
    else:
        ds_init = 'from .base import BaseDataset\n\nDATASET_REGISTRY = {}\n'

    write_if_changed(os.path.join(datasets_dst, '__init__.py'), ds_init)
    copied.append('datasets/__init__.py (generated)')

    # ================================================================
    # Output result
    # ================================================================
    print(json.dumps({
        'status': 'ok',
        'workspace_dir': dst,
        'model_name': model_name,
        'data_name': data_name,
        'model_package': pkg_name,
        'trainer': trainer_cls_name,
        'copied': copied,
    }))


if __name__ == '__main__':
    main()
