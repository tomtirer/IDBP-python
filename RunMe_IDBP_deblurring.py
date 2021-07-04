import os.path
import logging

import numpy as np
from collections import OrderedDict
from scipy import ndimage

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util
from utils_idbp import *


"""
%%  IDBP for (non-blind) deblurring

@article{tirer2018image,
  title={Image restoration by iterative denoising and backward projections},
  author={Tirer, Tom and Giryes, Raja},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={3},
  pages={1220--1234},
  year={2018},
  publisher={IEEE}
}

This is a python IDBP implementation (the original implementation was in matlab: https://github.com/tomtirer/IDBP)
It is not optimized for runtime (mostly using numpy) to facilitate using off-the-shelf denoisers that do not run on GPU/PyTorch.
It builds on the CNN denoisers and some code from github: https://github.com/cszn/DPIR

@article{zhang2020plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint},
  year={2020}
}

"""

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = 'ircnn_color'           # 'drunet_gray' | 'drunet_color' | 'ircnn_gray' | 'ircnn_color'
    testset_name = 'classics'               # test set,  'set5' | 'classics'
    iter_num = 20                         # number of iterations, 30 iterations used in IDBP paper, but often 15-20 are enough.

    perIter_print = False
    show_img = False                     # default: False
    save_L = True                        # save LR/blurred image
    save_E = True                        # save estimated image
    border = 0

    task_current = 'deblur'
    n_channels = 3 if 'color' in  model_name else 1  # fixed
    model_zoo = 'denoisers_folder/model_zoo'              # fixed
    testsets = 'testsets'                # fixed
    results = 'results'                  # fixed
    result_name = testset_name + '_' + task_current + '_IDBP_w_' + model_name + '_denoiser'
    model_path = os.path.join(model_zoo, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    if 'drunet' in model_name:
        from denoisers_folder.models.network_unet import UNetRes as net
        model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
    elif 'ircnn' in model_name:
        from denoisers_folder.models.network_dncnn import IRCNN as net
        model = net(in_nc=n_channels, out_nc=n_channels, nc=64)
        model25 = torch.load(model_path)
        former_idx = 0

    logger.info('model_name:{}'.format(model_name))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    test_results_ave = OrderedDict()
    test_results_ave['psnr'] = []  # record average PSNR for each kernel

    for scenario_ind in range(1,5):

        if scenario_ind == 1:
            noise_level_img = np.sqrt(2)/255.0
            k = np.zeros((15,15))
            for i in range(15):
                for j in range(15):
                    k[i, j] = 1. / ((i-7)**2 + (j-7)**2 + 1)
            k = k / k.sum()
            # Alg params:
            epsilon_alg = 4e-3 * (noise_level_img*255)**2
            sigma_alg = noise_level_img + 10.0/255.0  # per-iteration value will probably improve results

        elif scenario_ind == 2:
            noise_level_img = np.sqrt(8)/255.0
            k = np.zeros((15,15))
            for i in range(15):
                for j in range(15):
                    k[i, j] = 1. / ((i-7)**2 + (j-7)**2 + 1)
            k = k/k.sum()
            # Alg params:
            epsilon_alg = 2e-3 * (noise_level_img*255)**2
            sigma_alg = noise_level_img + 10.0/255.0  # per-iteration value will probably improve results

        elif scenario_ind == 3:
            noise_level_img = np.sqrt(0.3)/255.0
            k = np.ones((9,9))
            k = k / k.sum()
            # Alg params:
            epsilon_alg = 3e-3 * (noise_level_img*255)**2
            sigma_alg = noise_level_img + 10.0/255.0  # per-iteration value will probably improve results

        else:  # scenario_ind == 4:
            noise_level_img = np.sqrt(49)/255.0
            k = np.outer(np.array([1, 4, 6, 4, 1]), np.array([1, 4, 6, 4, 1]))
            k = k / k.sum()
            # Alg params:
            epsilon_alg = 0.8e-3 * (noise_level_img*255)**2
            sigma_alg = noise_level_img + 10.0/255.0  # per-iteration value will probably improve results

        logger.info('-------k:{:>2d} ---------'.format(scenario_ind))
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['isnr'] = []
        k = k.astype(np.float64)
        util.imshow(k) if show_img else None


        H_func = lambda Z: cconv2_by_fft2(Z,k,flag_invertB=0)
        Ht_func = lambda Z: cconv2_by_fft2(Z,np.fliplr(np.flipud(np.conj(k))), flag_invertB=0)  # LS gradient:  Ht_func( Y - H_func(X) )
        Hdagger_func = lambda Z: cconv2_by_fft2(Z,k,flag_invertB=1,epsilon=epsilon_alg)  # BP gradient:  Hdagger_func( Y - H_func(X) )

        for idx, img in enumerate(L_paths):

            # --------------------------------
            # prepare observations
            # --------------------------------

            img_name, ext = os.path.splitext(os.path.basename(img))
            img_H = util.imread_uint(img, n_channels=n_channels)
            #img_H = util.modcrop(img_H, 8)  # modcrop
            img_H = util.uint2single(img_H)
            M, N = img_H.shape[:2]

            # equivalently to: (from scipy import ndimage)
            img_L = ndimage.filters.convolve(img_H, np.expand_dims(k, axis=2), mode='wrap')
            # you can use:
            # img_L = np.zeros((M, N, n_channels))
            # for c in range(n_channels):
            #     img_L[:, :, c] =  H_func(img_H[:, :, c])
            util.imshow(img_L) if show_img else None

            np.random.seed(seed=0)  # for reproducibility
            img_L = img_L + np.random.normal(0, noise_level_img, img_L.shape)  # add AWGN

            input_psnr = util.calculate_psnr(255 * img_L, 255 * img_H, border=border)  # change with your own border

            # --------------------------------
            # run IDBP deblurring
            # --------------------------------

            Y = img_L
            X = Y  # initialization
            step_size = 1

            for i in range(iter_num):

                # --------------------------------
                # BP data fidelity gradient step
                # --------------------------------

                Z = np.zeros((M, N, n_channels))
                for c in range(n_channels):
                    resid = Y[:,:,c] - H_func(X[:,:,c])
                    Z[:,:,c] = X[:,:,c] + step_size*Hdagger_func(resid)

                Z_ = util.single2tensor4(Z).to(device)

                # --------------------------------
                # denoiser prior step
                # --------------------------------

                if 'drunet' in model_name:
                    sigma = torch.tensor(sigma_alg).to(device)
                    Z_ = torch.cat((Z_, sigma.float().repeat(1, 1, Z_.shape[2], Z_.shape[3])), dim=1)
                    X = utils_model.test_mode(model, Z_, mode=2, refield=32, min_size=256, modulo=16)

                elif 'ircnn' in model_name:
                    current_idx = np.int(np.ceil(sigma_alg * 255. / 2.) - 1)
                    if current_idx != former_idx:
                        model.load_state_dict(model25[str(current_idx)], strict=True)
                        model.eval()
                        for _, v in model.named_parameters():
                            v.requires_grad = False
                        model = model.to(device)
                    former_idx = current_idx
                    X = model(Z_)

                X = util.tensor2single(X)
                if X.ndim == 2:
                    X = np.expand_dims(X, axis=2)

                img_E = X
                if perIter_print:
                    psnr = util.calculate_psnr(255*img_E, 255*img_H, border=border)  # change with your own border
                    print('IDBP: finished iteration ({}), sigma_alg: {:.2f}, PSNR: {:.2f}dB'.format(i, sigma_alg, psnr))

            # --------------------------------
            # save estimated img_E
            # --------------------------------

            img_E = util.single2uint(img_E)
            img_H = util.single2uint(img_H)
            if n_channels == 1:
                img_E = img_E.squeeze()
                img_H = img_H.squeeze()

            if save_E:
                util.imsave(img_E, os.path.join(E_path, img_name+'_S'+str(scenario_ind)+'_IDBP_w_'+model_name+'_denoiser.png'))

            if save_L:
                util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name+'_S'+str(scenario_ind)+'_LR.png'))

            psnr = util.calculate_psnr(img_E, img_H, border=border)  # change with your own border
            test_results['psnr'].append(psnr)
            test_results['isnr'].append(psnr-input_psnr)
            logger.info('{:->4d}--> {:>10s} -- Scenario {}, ISNR (PSNR-InputPSNR): {:.2f}dB, PSNR: {:.2f}dB'.format(idx+1, img_name+ext, scenario_ind, psnr-input_psnr, psnr))

        # --------------------------------
        # Average PSNR
        # --------------------------------

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_isnr = sum(test_results['isnr']) / len(test_results['isnr'])
        logger.info('------> Average PSNR results for set: ({}), Scenario: ({}), sigma: ({:.2f}), ISNR(PSNR-InputPSNR): {:.2f}dB, PSNR: {:.2f} dB'.format(testset_name, scenario_ind, noise_level_img, ave_isnr, ave_psnr))
        test_results_ave['psnr'].append(ave_psnr)

if __name__ == '__main__':

    main()
