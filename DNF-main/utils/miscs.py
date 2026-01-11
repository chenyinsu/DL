import torch
import os
import shutil
import cv2
import numpy as np

def load_checkpoint(config, model, optimizer, lr_scheduler, logger, epoch=None):
    resume_ckpt_path = config['train']['resume']
    logger.info(f"==============> Resuming form {resume_ckpt_path}....................")
    if resume_ckpt_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            resume_ckpt_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(resume_ckpt_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_psnr = 0.0
    if not config.get('eval_mode', False) and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if 'max_psnr' in checkpoint:
            max_psnr = checkpoint['max_psnr']
    if epoch is None and 'epoch' in checkpoint:
        config['train']['start_epoch'] = checkpoint['epoch']
        logger.info(f"=> loaded successfully '{resume_ckpt_path}' (epoch {checkpoint['epoch']})")
    del checkpoint
    torch.cuda.empty_cache()
    return max_psnr

# def load_pretrained(config, model, logger):
#     logger.info(f"==============> Loading weight {config['train']['pretrained']}....................")
#     checkpoint = torch.load(config['train']['pretrained'], map_location='cpu')
#     state_dict = checkpoint['model']
#     msg = model.load_state_dict(state_dict, strict=False)
#     logger.warning(msg)
#
#     logger.info(f"=> loaded successfully '{config['train']['pretrained']}'")
#
#     del checkpoint
#     torch.cuda.empty_cache()


# def load_pretrained(config, model, logger):
#     logger.info(f"==============> Loading weight {config['train']['pretrained']}....................")
#
#     checkpoint = torch.load(config['train']['pretrained'], map_location='cpu')
#     state_dict = checkpoint['model']
#
#     # ====== 检查每个参数的 shape ======
#     model_dict = model.state_dict()
#     mismatch_count = 0
#     for k in model_dict.keys():
#         if k in state_dict:
#             if model_dict[k].shape != state_dict[k].shape:
#                 logger.warning(f"[MISMATCH] {k} | model: {model_dict[k].shape} | checkpoint: {state_dict[k].shape}")
#                 mismatch_count += 1
#         else:
#             logger.warning(f"[MISSING in checkpoint] {k}")
#
#     for k in state_dict.keys():
#         if k not in model_dict:
#             logger.warning(f"[EXTRA in checkpoint] {k}")
#
#     logger.info(f"Total mismatched parameters: {mismatch_count}")
#     # ==================================
#
#     # 加载权重
#     msg = model.load_state_dict(state_dict, strict=False)
#     logger.warning(msg)
#
#     logger.info(f"=> loaded successfully '{config['train']['pretrained']}'")
#
#     del checkpoint
#     torch.cuda.empty_cache()


def load_pretrained(config, model, logger, load_modules=None):
    """
    load_modules: list[str]，只加载这些模块的参数（例如 ["denoising_block", "aux_denoising_blocks"]）
                  如果为 None，表示加载所有兼容的参数
    """
    logger.info(f"==============> Loading pretrained weight {config['train']['pretrained']} ....................")
    checkpoint = torch.load(config['train']['pretrained'], map_location='cpu')
    pretrained_dict = checkpoint['model']
    model_dict = model.state_dict()

    # 根据指定模块过滤参数
    if load_modules is not None:
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if any(k.startswith(m) for m in load_modules)
        }

    # 只保留形状匹配的参数
    compatible_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}

    # 更新并加载
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)

    # 打印加载情况
    loaded_keys = set(compatible_dict.keys())
    total_keys = set(model_dict.keys())
    skipped_keys = total_keys - loaded_keys

    logger.info(f"=> Loaded {len(loaded_keys)} layers from pretrained model.")
    if skipped_keys:
        logger.warning(f"=> Skipped {len(skipped_keys)} layers (new or modified): {list(skipped_keys)[:10]} ...")

    # 冻结已加载的层
    for name, param in model.named_parameters():
        if name in loaded_keys:
            param.requires_grad = False
    logger.info("=> Frozen pretrained layers. Only training new/modified modules.")

    del checkpoint
    torch.cuda.empty_cache()



def save_checkpoint(config, epoch, model, max_psnr, optimizer, lr_scheduler, logger, is_best=False):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_psnr': max_psnr,
                  'epoch': epoch,
                  'config': config}

    os.makedirs(os.path.join(config['output'], 'checkpoints'), exist_ok=True)

    save_path = os.path.join(config['output'], 'checkpoints', 'checkpoint.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved")
    if epoch % config['save_per_epoch'] == 0 or (config['train']['epochs'] - epoch) < 50:
        shutil.copy(save_path, os.path.join(config['output'], 'checkpoints', f'epoch_{epoch:04d}.pth'))
        logger.info(f"{save_path} copied to epoch_{epoch:04d}.pth")
    if is_best:
        shutil.copy(save_path, os.path.join(config['output'], 'checkpoints', 'model_best.pth'))
        logger.info(f"{save_path} copied to model_best.pth")


'''

def save_checkpoint(config, epoch, model, max_psnr, optimizer, lr_scheduler, logger, is_best=False):
    """
    仅保存当前最优模型的检查点（checkpoint），避免存储不必要的模型文件以节省空间。

    参数：
    - config: 训练的配置信息（字典）。
    - epoch: 当前训练的轮数。
    - model: 训练的神经网络模型。
    - max_psnr: 最高 PSNR 值（用于图像质量评估）。
    - optimizer: 训练所用的优化器。
    - lr_scheduler: 学习率调度器。
    - logger: 日志记录工具。
    - is_best: 是否是当前最优模型（默认为 False）。
    """

    # 仅在当前模型是最优模型时保存
    if is_best:
        save_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'max_psnr': max_psnr,
            'epoch': epoch,
            'config': config
        }

        # 确保 `checkpoints/` 目录存在
        checkpoints_dir = os.path.join(config['output'], 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        # 定义最优模型的保存路径
        best_model_path = os.path.join(checkpoints_dir, 'model_best.pth')

        # 记录日志
        logger.info(f"Saving best model to {best_model_path} ...")

        # 保存最优模型
        torch.save(save_state, best_model_path)

        logger.info(f"Best model saved at {best_model_path}")
'''

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def save_image_torch(img, file_path, range_255_float=True, params=None, auto_mkdir=True):
    """Write image to file.
    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.
    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)

    assert len(img.size()) == 3
    img = img.clone().cpu().detach().numpy().transpose(1, 2, 0)

    if range_255_float:
        # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
        img = img.clip(0, 255).round()
        img = img.astype(np.uint8)
    else:
        img = img.clip(0, 1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')