import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')
import os
data_pth = r'/media/yl/yl_8t/bvqa360_experiments/test_dmos'
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models 加载预训练模型
        load_path = self.opt['path'].get('pretrain_network_g', None)
        #print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!!!!!!!!!!!!!!!!!!!!!!!!!!!!', load_path)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    # 加载训练数据
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        #print('The lq is %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%', self.lq.size())
        # save_im = self.lq[0, 0, :, :, :].permute(1,2,0).detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.imsave(r'/media/yl/yl_8t/data_bvqa_lc_resize/im1.jpg', save_im)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            #print('The gt is %%%%%%%%%%%%%%%%%%%%%%%%%%%', self.gt)

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq) # b,t,c,h,w
        tb_logger.add_images('b_t_image', self.lq.view(self.lq.size(0)*self.lq.size(1), self.lq.size(2),
                                         self.lq.size(3), self.lq.size(4)), global_step=current_iter)

        # for i in range(self.lq.size(0)):
        #     for j in range(self.lq.size(1)):
        #         tb_logger.add_image('image_'+str(i) + '_' +str(j), self.lq[i][j], global_step=current_iter)


        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>The training score--DMOS is', self.output.cpu().detach(), self.gt.cpu().detach())
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict) # 多GPU训练

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']

        pbar = tqdm(total=len(dataloader), unit='image')
        data_pth = dataloader.dataset.opt['result_pth']
        predicted_dmos = dataloader.dataset.opt['test_dmos']
        f1 = open(os.path.join(data_pth, predicted_dmos), 'w')


        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            #sr_img = tensor2img([visuals['result']])
            # if 'gt' in visuals:
            #     gt_img = tensor2img([visuals['gt']])
            #     del self.gt
            im_name = ('_').join(img_name.split('_')[:-1])
            frame_id = img_name.split('_')[-1]
            pre_dmos = visuals['result']
            gt_dmo = visuals['gt']

            print(">>>>>>>>>>>>>>Vid {}: pred {} dmos {}<<<<<<<<<<<<<<".format(im_name, pre_dmos.detach().item(), gt_dmo.detach().item()))
            line1 = '{} {} {} {}\n'.format(im_name, frame_id, pre_dmos.detach().item(), gt_dmo.detach().item())
            f1.writelines(line1)



            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        f1.close()
        pbar.close()

    # def nondist_validation(self, dataloader, current_iter, tb_logger,
    #                        save_img):
    #     dataset_name = dataloader.dataset.opt['name']
    #
    #     pbar = tqdm(total=len(dataloader), unit='image')
    #
    #     f1 = open(os.path.join(data_pth, 'BVQA360_dmos.txt'), 'w')
    #
    #
    #     for idx, val_data in enumerate(dataloader):
    #         img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
    #         self.feed_data(val_data)
    #         self.test()
    #
    #         visuals = self.get_current_visuals()
    #         #sr_img = tensor2img([visuals['result']])
    #         # if 'gt' in visuals:
    #         #     gt_img = tensor2img([visuals['gt']])
    #         #     del self.gt
    #         im_name = ('_').join(img_name.split('_')[:-1])
    #         frame_id = img_name.split('_')[-1]
    #         pre_dmos = visuals['result']
    #         gt_dmo = visuals['gt']
    #
    #
    #         line1 = '{} {} {} {}\n'.format(im_name, frame_id, pre_dmos, gt_dmo)
    #         f1.writelines(line1)
    #
    #
    #
    #         # tentative for out of GPU memory
    #         del self.lq
    #         del self.output
    #         torch.cuda.empty_cache()
    #
    #         pbar.update(1)
    #         pbar.set_description(f'Test {img_name}')
    #     f1.close()
    #     pbar.close()

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
