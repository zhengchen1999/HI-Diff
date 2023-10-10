import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from hi_diff.utils.base_model import BaseModel
from torch.nn import functional as F
from functools import partial
import numpy as np
from hi_diff.utils.beta_schedule import make_beta_schedule, default
from ldm.ddpm import DDPM


@MODEL_REGISTRY.register()
class HI_Diff_S2(BaseModel):
    """HI-Diff model for test."""

    def __init__(self, opt):
        super(HI_Diff_S2, self).__init__(opt)

        # define network
        self.net_le = build_network(opt['network_le'])
        self.net_le = self.model_to_device(self.net_le)
        self.print_network(self.net_le)

        self.net_le_dm = build_network(opt['network_le_dm'])
        self.net_le_dm = self.model_to_device(self.net_le_dm)
        self.print_network(self.net_le_dm)

        self.net_d = build_network(opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_le', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_le', 'params')
            self.load_network(self.net_le, load_path, self.opt['path'].get('strict_load_le', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_le_dm', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_le_dm', 'params')
            self.load_network(self.net_le_dm, load_path, self.opt['path'].get('strict_load_le_dm', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # diffusion
        self.apply_ldm = self.opt['diffusion_schedule'].get('apply_ldm', None)
        if self.apply_ldm:
            # apply LDM implementation
            self.diffusion = DDPM(denoise=self.net_d, 
                                  condition=self.net_le_dm, 
                                  n_feats=opt['network_g']['embed_dim'], 
                                  group=opt['network_g']['group'],
                                  linear_start= self.opt['diffusion_schedule']['linear_start'],
                                  linear_end= self.opt['diffusion_schedule']['linear_end'], 
                                  timesteps = self.opt['diffusion_schedule']['timesteps'])
            self.diffusion = self.model_to_device(self.diffusion)
        else:
            # implemented locally
            self.set_new_noise_schedule(self.opt['diffusion_schedule'], self.device)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        self.net_d.train()
        self.net_le.train()
        self.net_le_dm.train()
        if self.apply_ldm:
            self.diffusion.train()
        
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            print("TODO")

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.cri_pix_diff = build_loss(train_opt['pixel_diff_opt']).to(self.device)
        else:
            self.cri_pix = None
            self.cri_pix_diff = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
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
                logger.warning(f'Network G: Params {k} will not be optimized.')

        if self.apply_ldm:
            for k, v in self.diffusion.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Network Diffusion: Params {k} will not be optimized.')
        else:
            for k, v in self.net_le_dm.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Network LE-DM: Params {k} will not be optimized.')

            for k, v in self.net_d.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Network D: Params {k} will not be optimized.')

        optim_type = train_opt['optim_total'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_total = torch.optim.Adam(optim_params, **train_opt['optim_total'])
        elif optim_type == 'AdamW':
            self.optimizer_total = torch.optim.AdamW(optim_params, **train_opt['optim_total'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_total)

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        # β1, β2, ..., βΤ (T)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['timesteps'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        # α1, α2, ..., αΤ (T)
        alphas = 1. - betas
        # α1, α1α2, ..., α1α2...αΤ (T)
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # 1, α1, α1α2, ...., α1α2...αΤ-1 (T)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        # 1, √α1, √α1α2, ...., √α1α2...αΤ (T+1)
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        if condition_x is None:
            raise RuntimeError('Must have LQ/LR condition')

        if ema_model:
            print("TODO")
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.net_d(x, condition_x, torch.full(x.shape, t+1, device=self.betas.device, dtype=torch.long)))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance
    
    def p_sample_wo_variance(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        model_mean, _ = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, ema_model=ema_model)
        return model_mean
    
    def p_sample_loop_wo_variance(self, x_in, x_noisy, ema_model=False):
        img = x_noisy
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample_wo_variance(img, i, condition_x=x_in, ema_model=ema_model)
        return img

    def p_sample(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        model_mean, _ = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, ema_model=ema_model)
        return model_mean

    def p_sample_loop(self, x_in, x_noisy, ema_model=False):
        img = x_noisy
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, i, condition_x=x_in, ema_model=ema_model)
        return img

    def q_sample(self, x_start, sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            sqrt_alpha_cumprod * x_start +
            (1 - sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter, noise=None):
        # freeze c1 (cpen_s1)
        for p in self.net_le.parameters():
            p.requires_grad = False
        
        self.optimizer_total.zero_grad()
        prior_z = self.net_le(self.lq, self.gt)

        if self.apply_ldm:
            prior, _=self.diffusion(self.lq, prior_z)
        else:
            prior_d = self.net_le_dm(self.lq)
            # diffusion-forward
            t = self.opt['diffusion_schedule']['timesteps']
            # [b, 4c']
            noise = default(noise, lambda: torch.randn_like(prior_z))
            # sample xt/x_noisy (from x0/x_start)
            prior_noisy = self.q_sample(
                x_start=prior_z, sqrt_alpha_cumprod=self.alphas_cumprod[t-1],
                noise=noise)
            # diffusion-reverse
            prior = self.p_sample_loop_wo_variance(prior_d, prior_noisy)

        # ir
        self.output = self.net_g(self.lq, prior)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_pix_diff:
            l_pix_diff = self.cri_pix_diff(prior_z, prior)
            l_total += l_pix_diff
            loss_dict['l_pix_diff'] = l_pix_diff

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
        if self.opt['train']['use_grad_clip']:
            if self.apply_ldm:
                torch.nn.utils.clip_grad_norm_(list(self.net_g.parameters()) + list(self.diffusion.parameters()), 0.01)
            else:
                torch.nn.utils.clip_grad_norm_(list(self.net_g.parameters()) + list(self.net_le_dm.parameters()) + list(self.net_d.parameters()), 0.01)
        self.optimizer_total.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        scale = self.opt.get('scale', 1)
        window_size = 8
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        if hasattr(self, 'net_g_ema'):
            print("TODO")
        else:
            if self.apply_ldm:
                self.net_g.eval()
                self.diffusion.eval()

                with torch.no_grad():
                    prior = self.diffusion(img)
                    self.output = self.net_g(img, prior)
                    self.diffusion.train()

                self.net_g.train()
                self.diffusion.train()
            else:
                self.net_le.eval()
                self.net_le_dm.eval()
                self.net_d.eval()
                self.net_g.eval()

                with torch.no_grad():
                    prior_c = self.net_le_dm(img)
                    prior_noisy = torch.randn_like(prior_c)
                    prior = self.p_sample_loop(prior_c, prior_noisy)
                    self.output = self.net_g(img, prior)

                self.net_le.train()
                self.net_le_dm.train()
                self.net_d.train()
                self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')

                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            print("TODO")
        else:
            if self.apply_ldm:
                if self.opt['dist']:
                    self.net_le_dm = self.diffusion.module.condition
                    self.net_d = self.diffusion.module.model
                else:
                    self.net_le_dm = self.diffusion.condition
                    self.net_d = self.diffusion.model
            self.save_network(self.net_g, 'net_g', current_iter)
            self.save_network(self.net_le_dm, 'net_le_dm', current_iter)
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
