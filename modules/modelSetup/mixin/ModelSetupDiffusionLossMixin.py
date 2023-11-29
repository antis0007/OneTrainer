from abc import ABCMeta

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.module.AestheticScoreModel import AestheticScoreModel
from modules.module.HPSv2ScoreModel import HPSv2ScoreModel
from modules.util import loss_util
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.AlignPropLoss import AlignPropLoss
#Sourced from: https://zhuanlan.zhihu.com/p/648859856
def compute_snr(timesteps, alphas_cumprod):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)
        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

class ModelSetupDiffusionLossMixin(metaclass=ABCMeta):
    def __init__(self):
        super(ModelSetupDiffusionLossMixin, self).__init__()
        self.align_prop_loss_fn = None

    def _get_lora_rank(
            self,
            state_dict: dict,
    ) -> int:
        for name, state in state_dict.items():
            if name.endswith("lora_down.weight"):
                return state.shape[0]

    def _diffusion_loss(
            self,
            model: StableDiffusionXLModel,
            batch: dict,
            data: dict,
            args: TrainArgs,
            train_device: torch.device,
    ) -> Tensor:
        if data['loss_type'] == 'align_prop':
            if self.align_prop_loss_fn is None:
                dtype = data['predicted'].dtype

                match args.align_prop_loss:
                    case AlignPropLoss.HPS:
                        self.align_prop_loss_fn = HPSv2ScoreModel(dtype)
                    case AlignPropLoss.AESTHETIC:
                        self.align_prop_loss_fn = AestheticScoreModel()

                self.align_prop_loss_fn.to(device=train_device, dtype=dtype)
                self.align_prop_loss_fn.requires_grad_(False)
                self.align_prop_loss_fn.eval()

            match args.align_prop_loss:
                case AlignPropLoss.HPS:
                    with torch.autocast(device_type=train_device.type, dtype=data['predicted'].dtype):
                        losses = self.align_prop_loss_fn(data['predicted'], batch['prompt'], train_device)
                case AlignPropLoss.AESTHETIC:
                    losses = self.align_prop_loss_fn(data['predicted'])

            losses = losses * args.align_prop_weight
        else:
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if args.masked_training and not args.model_type.has_conditioning_image_input():
                losses = loss_util.masked_loss(
                    F.mse_loss,
                    data['predicted'],
                    data['target'],
                    batch['latent_mask'],
                    args.unmasked_weight,
                    args.normalize_masked_area_loss
                ).mean([1, 2, 3])
            else:
                if (hasattr(model, "noise_scheduler") and args.debiased_loss == True):
                    # timesteps = torch.randint(
                    #     low=0,
                    #     high=int(model.noise_scheduler.config['num_train_timesteps'] * args.max_noising_strength),
                    #     size=(data['predicted'].shape[0],),
                    #     device=args.train_device,
                    # ).long()
                    #Training Details. We set T = 1000 for all experiments. We implement the proposed approach
                    #on top of ADM (Dhariwal & Nichol, 2021), which offers well-designed architecture and efficient
                    #sampling. We train our model for 500K iterations with a batch size of 8
                    # Compute SNR
                    #timestep = model.noise_scheduler.timesteps[-1]
                    timestep = 999 #[0-999] = 1000 timesteps
                    alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(args.train_device)
                    alpha_prod = alphas_cumprod[timestep]
                    snr = alpha_prod / (1 - alpha_prod)

                    #timesteps = model.noise_scheduler.timesteps
                    #alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(args.train_device)
                    #snr = compute_snr(timesteps, alphas_cumprod)
                    weight = 1 / torch.sqrt(snr)
                    # Apply weighted loss
                    mse_loss = F.mse_loss(
                        data['predicted'],
                        data['target'],
                        reduction='none'
                    ).mean([1, 2, 3])
                    losses = weight * mse_loss
                else:
                    losses = F.mse_loss(
                        data['predicted'],
                        data['target'],
                        reduction='none'
                    ).mean([1, 2, 3])

                if args.masked_training and args.normalize_masked_area_loss:
                    clamped_mask = torch.clamp(batch['latent_mask'], args.unmasked_weight, 1)
                    losses = losses / clamped_mask.mean(dim=(1, 2, 3))

        return losses.mean()
