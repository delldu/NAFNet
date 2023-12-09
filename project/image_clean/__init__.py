"""Video Matte Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import todos

from . import NAFNet_arch

import pdb


def get_denoise_model():
    """Create model."""
    device = todos.model.get_device()

    # SIDD/NAFNet-width64.yml
    # network_g:
    #   type: NAFNet
    #   width: 64
    #   enc_blk_nums: [2, 2, 4, 8]
    #   middle_blk_num: 12
    #   dec_blk_nums: [2, 2, 2, 2]
    model = NAFNet_arch.NAFNet(width=64, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])

    model.load_weights("models/image_denoise.pth")
    # model = todos.model.ResizePadModel(model)
    model = model.to(device)
    model.eval()
    if 'cpu' in str(device.type):
        model.float()

    print(f"Running on {device} ...")

    # # make sure model good for C/C++
    # model = torch.jit.script(model)
    # # https://github.com/pytorch/pytorch/issues/52286
    # torch._C._jit_set_profiling_executor(False)
    # # C++ Reference
    # # torch::jit::getProfilingMode() = false;                                                                                                             
    # # torch::jit::setTensorExprFuserEnabled(false);

    # # print(model.code)
    # todos.data.mkdir("output")
    # if not os.path.exists("output/image_denoise.torch"):
    #     model.save("output/image_denoise.torch")

    return model, device


def get_deblur_model():
    """Create model."""
    device = todos.model.get_device()

    # REDS/NAFNet-width64.yml
    # network_g:
    #   type: NAFNetLocal
    #   width: 64
    #   enc_blk_nums: [1, 1, 1, 28]
    #   middle_blk_num: 1
    #   dec_blk_nums: [1, 1, 1, 1]

    args={}
    kwargs={'width': 64, 'enc_blk_nums': [1, 1, 1, 28], 'middle_blk_num': 1, 'dec_blk_nums': [1, 1, 1, 1]}
    model = NAFNet_arch.NAFNetLocal(**args, **kwargs)

    model.load_weights("models/image_deblur.pth")
    # model = todos.model.ResizePadModel(model)
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    # # make sure model good for C/C++
    # model = torch.jit.script(model)
    # # https://github.com/pytorch/pytorch/issues/52286
    # torch._C._jit_set_profiling_executor(False)
    # # C++ Reference
    # # torch::jit::getProfilingMode() = false;                                                                                                             
    # # torch::jit::setTensorExprFuserEnabled(false);

    # todos.data.mkdir("output")
    # if not os.path.exists("output/image_deblur.torch"):
    #     model.save("output/image_deblur.torch")

    return model, device


def denoise_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_denoise_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        B, C, H, W = input_tensor.size()
        # input_tensor += 0.25 * torch.randn(B, C, H, W).clamp(0.0, 1.0)

        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()

        predict_tensor = todos.model.forward(model, device, input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()


def denoise_add_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_denoise_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        B, C, H, W = input_tensor.size()
        input_tensor += 0.15 * torch.randn(B, C, H, W).clamp(0.0, 1.0)

        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()

        predict_tensor = todos.model.forward(model, device, input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()



def deblur_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_deblur_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()

        predict_tensor = todos.model.forward(model, device, input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()
