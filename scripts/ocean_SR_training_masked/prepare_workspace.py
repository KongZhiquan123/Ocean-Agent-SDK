#!/usr/bin/env python3
"""
@filename prepare_workspace.py
@author kongzhiquan
@date 2026-02-11
@description 准备训练工作空间：只复制与所选模型相关的代码文件。

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
import re
import shutil
import sys


def get_package_name(cls):
    """获取一个类所在的包目录名（相对于上级包）"""
    src_file = inspect.getfile(cls)
    return os.path.basename(os.path.dirname(src_file))


def strip_header_docstring(content):
    """去除 Python 文件开头的模块级 docstring（保留 shebang 行）"""
    pattern = r'^((?:#!.*\n)?)\s*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')\s*\n?'
    return re.sub(pattern, r'\1', content, count=1)


def _is_up_to_date(src, dst):
    """判断目标文件是否已存在且不比源文件旧"""
    if not os.path.exists(dst):
        return False
    return os.path.getmtime(dst) >= os.path.getmtime(src)


def copy_file(src, dst):
    """复制文件，跳过已存在且未过期的文件。对 .py 文件去除头部 docstring。"""
    if _is_up_to_date(src, dst):
        return False  # 跳过，无需复制
    if src.endswith('.py'):
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        content = strip_header_docstring(content)
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(content)
        shutil.copystat(src, dst)
    else:
        shutil.copy2(src, dst)
    return True  # 实际执行了复制


def write_if_changed(path, content):
    """仅在内容有变化时写入文件"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            if f.read() == content:
                return False
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return True


def copy_dir(src, dst):
    """增量复制目录，仅复制新增或更新的文件，保留已有且未变化的文件"""
    for dirpath, dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        dst_dir = os.path.join(dst, rel)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in filenames:
            copy_file(os.path.join(dirpath, fname),
                      os.path.join(dst_dir, fname))


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
        'generate_training_plots.py',
        'validate_dataset.py', 'check_gpu.py', 'list_models.py',
        'estimate_memory.py', 'check_output_shape.py',
    ]
    for f in core_files:
        s = os.path.join(src, f)
        if os.path.exists(s):
            copy_file(s, os.path.join(dst, f))
            copied.append(f)

    # ================================================================
    # 2. utils/ 和 template_configs/（完整复制，体积小且通用）
    # ================================================================
    for d in ['utils', 'template_configs', 'training_plot_lib']:
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
    os.makedirs(models_dst, exist_ok=True)

    if is_resshift:
        copy_dir(os.path.join(src, 'models', 'resshift'),
                 os.path.join(models_dst, 'resshift'))
        copied.append('models/resshift/')
        init_code = (
            'from . import resshift\n\n'
            '_model_dict = {}\n\n'
            '_ddpm_dict = {}\n'
        )
        pkg_name = "resshift"
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

    write_if_changed(os.path.join(models_dst, '__init__.py'), init_code)
    copied.append('models/__init__.py (generated)')

    # ================================================================
    # 4. trainers/：base.py + 模型对应的 trainer
    # ================================================================
    trainer_cls = _trainer_dict[model_name]
    trainer_cls_name = trainer_cls.__name__
    trainer_file = inspect.getfile(trainer_cls)
    trainer_basename = os.path.basename(trainer_file)

    trainers_dst = os.path.join(dst, 'trainers')
    os.makedirs(trainers_dst, exist_ok=True)

    # base.py 始终需要（所有 trainer 都继承它）
    copy_file(os.path.join(src, 'trainers', 'base.py'),
              os.path.join(trainers_dst, 'base.py'))
    copied.append('trainers/base.py')

    # 如果不是 BaseTrainer 本身，复制专属 trainer
    trainer_init_lines = ['from .base import BaseTrainer']
    if trainer_basename != 'base.py':
        copy_file(trainer_file, os.path.join(trainers_dst, trainer_basename))
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
    write_if_changed(os.path.join(trainers_dst, '__init__.py'),
                      '\n'.join(trainer_init_lines))
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
    os.makedirs(forecastors_dst, exist_ok=True)

    copy_file(os.path.join(src, 'forecastors', 'base.py'),
              os.path.join(forecastors_dst, 'base.py'))
    copied.append('forecastors/base.py')

    fc_init_lines = ['from .base import BaseForecaster']
    for fc_file in _forecaster_extra.get(trainer_cls_name, []):
        fc_src = os.path.join(src, 'forecastors', fc_file)
        if os.path.exists(fc_src):
            copy_file(fc_src, os.path.join(forecastors_dst, fc_file))
            copied.append(f'forecastors/{fc_file}')
            fc_module = fc_file.replace('.py', '')
            if fc_module in _fc_cls_map:
                fc_init_lines.append(
                    f'from .{fc_module} import {_fc_cls_map[fc_module]}')

    write_if_changed(os.path.join(forecastors_dst, '__init__.py'),
                      '\n'.join(fc_init_lines) + '\n')
    copied.append('forecastors/__init__.py (generated)')

    # ================================================================
    # 6. datasets/：只复制选中的数据集
    # ================================================================
    datasets_dst = os.path.join(dst, 'datasets')
    os.makedirs(datasets_dst, exist_ok=True)

    if data_name in _dataset_dict:
        ds_cls = _dataset_dict[data_name]
        ds_file = inspect.getfile(ds_cls)
        ds_basename = os.path.basename(ds_file)
        ds_cls_name = ds_cls.__name__
        ds_module = ds_basename.replace('.py', '')

        copy_file(ds_file, os.path.join(datasets_dst, ds_basename))
        copied.append(f'datasets/{ds_basename}')

        ds_init = (
            f'from .{ds_module} import {ds_cls_name}\n\n'
            f'_dataset_dict = {{\n'
            f'    "{data_name}": {ds_cls_name},\n'
            f'}}\n'
        )
    else:
        ds_init = '_dataset_dict = {}\n'

    write_if_changed(os.path.join(datasets_dst, '__init__.py'), ds_init)
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
