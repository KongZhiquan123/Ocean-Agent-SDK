#!/usr/bin/env python3
"""
prepare_workspace.py

准备训练工作空间：只复制与所选模型相关的代码文件。
每次调用会清理 models/ trainers/ datasets/ forecastors/，
确保切换模型时旧代码被替换为新模型的代码。

用法:
    python prepare_workspace.py \
        --source_dir /path/to/ocean_SR_training_masked \
        --target_dir /path/to/workspace \
        --model_name SwinIR \
        --data_name OceanNPY
"""
import argparse
import inspect
import json
import os
import shutil
import sys


def get_package_name(cls):
    """获取一个类所在的包目录名（相对于上级包）"""
    src_file = inspect.getfile(cls)
    return os.path.basename(os.path.dirname(src_file))


def copy_dir(src, dst):
    """复制目录，已存在则先删除"""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    parser = argparse.ArgumentParser(description='准备训练工作空间')
    parser.add_argument('--source_dir', required=True, help='训练框架源目录')
    parser.add_argument('--target_dir', required=True, help='工作空间目标目录')
    parser.add_argument('--model_name', required=True, help='模型名称')
    parser.add_argument('--data_name', default='OceanNPY', help='数据集名称')
    args = parser.parse_args()

    src = os.path.abspath(args.source_dir)
    dst = os.path.abspath(args.target_dir)
    model_name = args.model_name
    data_name = args.data_name

    # 将源目录加入 sys.path 以导入注册表
    sys.path.insert(0, src)

    from models import _model_dict, _ddpm_dict
    from trainers import _trainer_dict
    from datasets import _dataset_dict

    is_resshift = model_name in {'Resshift', 'ResShift'}

    if not is_resshift and model_name not in _model_dict:
        print(json.dumps({
            'status': 'error',
            'error': f'未知模型: {model_name}',
            'available': list(_model_dict.keys()),
        }))
        sys.exit(1)

    if model_name not in _trainer_dict:
        print(json.dumps({
            'status': 'error',
            'error': f'模型 {model_name} 没有对应的 trainer',
        }))
        sys.exit(1)

    copied = []

    # ================================================================
    # 1. 核心文件（入口脚本、工具脚本）
    # ================================================================
    os.makedirs(dst, exist_ok=True)

    core_files = [
        'main.py', 'main_ddp.py', 'config.py',
        'generate_config.py', 'generate_training_report.py',
        'validate_dataset.py', 'check_gpu.py', 'list_models.py',
        'estimate_memory.py', 'check_output_shape.py',
    ]
    for f in core_files:
        s = os.path.join(src, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst, f))
            copied.append(f)

    # ================================================================
    # 2. utils/ 和 template_configs/（完整复制，体积小且通用）
    # ================================================================
    for d in ['utils', 'template_configs']:
        s = os.path.join(src, d)
        t = os.path.join(dst, d)
        if os.path.isdir(s):
            copy_dir(s, t)
            copied.append(f'{d}/')

    # ================================================================
    # 3. models/：只复制选中模型的包目录
    # ================================================================
    model_entry = _model_dict.get(model_name)
    is_diffusion = isinstance(model_entry, dict)

    models_dst = os.path.join(dst, 'models')
    if os.path.exists(models_dst):
        shutil.rmtree(models_dst)
    os.makedirs(models_dst)

    if is_resshift:
        copy_dir(os.path.join(src, 'models', 'resshift'),
                 os.path.join(models_dst, 'resshift'))
        copied.append('models/resshift/')
        init_code = (
            'from . import resshift\n\n'
            '_model_dict = {}\n\n'
            '_ddpm_dict = {}\n'
        )
    elif is_diffusion:
        # 扩散模型：entry = {"model": ddpm.UNet, "diffusion": ddpm.GaussianDiffusion}
        model_cls = model_entry['model']
        diff_cls = model_entry['diffusion']
        pkg_name = get_package_name(model_cls)
        model_cls_name = model_cls.__name__
        diff_cls_name = diff_cls.__name__

        copy_dir(os.path.join(src, 'models', pkg_name),
                 os.path.join(models_dst, pkg_name))
        copied.append(f'models/{pkg_name}/')

        # 生成最小 __init__.py
        init_code = (
            f'from . import {pkg_name}\n\n'
            f'_model_dict = {{\n'
            f'    "{model_name}": {{"model": {pkg_name}.{model_cls_name}, '
            f'"diffusion": {pkg_name}.{diff_cls_name}}},\n'
            f'}}\n\n'
            f'_ddpm_dict = {{\n'
            f'    "{model_name}": {{"model": {pkg_name}.{model_cls_name}, '
            f'"diffusion": {pkg_name}.{diff_cls_name}}},\n'
            f'}}\n'
        )
    else:
        # 标准模型：entry = SwinIR_net (class)
        cls_name = model_entry.__name__
        pkg_name = get_package_name(model_entry)

        copy_dir(os.path.join(src, 'models', pkg_name),
                 os.path.join(models_dst, pkg_name))
        copied.append(f'models/{pkg_name}/')

        init_code = (
            f'from .{pkg_name} import {cls_name}\n\n'
            f'_model_dict = {{\n'
            f'    "{model_name}": {cls_name},\n'
            f'}}\n\n'
            f'_ddpm_dict = {{}}\n'
        )

    with open(os.path.join(models_dst, '__init__.py'), 'w') as f:
        f.write(init_code)
    copied.append('models/__init__.py (generated)')

    # ================================================================
    # 4. trainers/：base.py + 模型对应的 trainer
    # ================================================================
    trainer_cls = _trainer_dict[model_name]
    trainer_cls_name = trainer_cls.__name__
    trainer_file = inspect.getfile(trainer_cls)
    trainer_basename = os.path.basename(trainer_file)

    trainers_dst = os.path.join(dst, 'trainers')
    if os.path.exists(trainers_dst):
        shutil.rmtree(trainers_dst)
    os.makedirs(trainers_dst)

    # base.py 始终需要（所有 trainer 都继承它）
    shutil.copy2(os.path.join(src, 'trainers', 'base.py'),
                 os.path.join(trainers_dst, 'base.py'))
    copied.append('trainers/base.py')

    # 如果不是 BaseTrainer 本身，复制专属 trainer
    trainer_init_lines = ['from .base import BaseTrainer']
    if trainer_basename != 'base.py':
        shutil.copy2(trainer_file, os.path.join(trainers_dst, trainer_basename))
        copied.append(f'trainers/{trainer_basename}')
        trainer_module = trainer_basename.replace('.py', '')
        trainer_init_lines.append(f'from .{trainer_module} import {trainer_cls_name}')

    trainer_init_lines += [
        '',
        '_trainer_dict = {',
        f'    "{model_name}": {trainer_cls_name},',
        '}',
        '',
    ]
    with open(os.path.join(trainers_dst, '__init__.py'), 'w') as f:
        f.write('\n'.join(trainer_init_lines))
    copied.append('trainers/__init__.py (generated)')

    # ================================================================
    # 5. forecastors/：base.py + 模型对应的 forecaster
    # ================================================================
    # trainer 类型 → 需要额外复制的 forecaster 文件
    _forecaster_extra = {
        'BaseTrainer': [],
        'DDPMTrainer': ['ddpm.py'],
        'ResshiftTrainer': ['resshift.py'],
        'ReMiGTrainer': ['ddpm.py'],
    }
    _fc_cls_map = {
        'ddpm': 'DDPMForecaster',
        'resshift': 'ResshiftForecaster',
    }

    forecastors_dst = os.path.join(dst, 'forecastors')
    if os.path.exists(forecastors_dst):
        shutil.rmtree(forecastors_dst)
    os.makedirs(forecastors_dst)

    shutil.copy2(os.path.join(src, 'forecastors', 'base.py'),
                 os.path.join(forecastors_dst, 'base.py'))
    copied.append('forecastors/base.py')

    fc_init_lines = ['from .base import BaseForecaster']
    for fc_file in _forecaster_extra.get(trainer_cls_name, []):
        fc_src = os.path.join(src, 'forecastors', fc_file)
        if os.path.exists(fc_src):
            shutil.copy2(fc_src, os.path.join(forecastors_dst, fc_file))
            copied.append(f'forecastors/{fc_file}')
            fc_module = fc_file.replace('.py', '')
            if fc_module in _fc_cls_map:
                fc_init_lines.append(
                    f'from .{fc_module} import {_fc_cls_map[fc_module]}')

    with open(os.path.join(forecastors_dst, '__init__.py'), 'w') as f:
        f.write('\n'.join(fc_init_lines) + '\n')
    copied.append('forecastors/__init__.py (generated)')

    # ================================================================
    # 6. datasets/：只复制选中的数据集
    # ================================================================
    datasets_dst = os.path.join(dst, 'datasets')
    if os.path.exists(datasets_dst):
        shutil.rmtree(datasets_dst)
    os.makedirs(datasets_dst)

    if data_name in _dataset_dict:
        ds_cls = _dataset_dict[data_name]
        ds_file = inspect.getfile(ds_cls)
        ds_basename = os.path.basename(ds_file)
        ds_cls_name = ds_cls.__name__
        ds_module = ds_basename.replace('.py', '')

        shutil.copy2(ds_file, os.path.join(datasets_dst, ds_basename))
        copied.append(f'datasets/{ds_basename}')

        ds_init = (
            f'from .{ds_module} import {ds_cls_name}\n\n'
            f'_dataset_dict = {{\n'
            f'    "{data_name}": {ds_cls_name},\n'
            f'}}\n'
        )
    else:
        ds_init = '_dataset_dict = {}\n'

    with open(os.path.join(datasets_dst, '__init__.py'), 'w') as f:
        f.write(ds_init)
    copied.append('datasets/__init__.py (generated)')

    # ================================================================
    # 输出结果
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
