import sys
import os
# from tabnanny import check
sys.path.insert(0, os.path.abspath('.'))


import torch
import torchvision.transforms as transforms
import json
import numpy as np
import copy
import pickle

from functools import partial

from ZSSGAN.model.sg2_model import Generator, Discriminator
from ZSSGAN.criteria.clip_loss import CLIPLoss       
import ZSSGAN.legacy as legacy
from spi.utils.load_utils import load_eg3d
from camera_utils import cal_canonical_c

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class EG3DGenerator(torch.nn.Module):
    def __init__(self, checkpoint_path):
        super(EG3DGenerator, self).__init__()

        # with open(checkpoint_path, 'rb') as f:
        #     if checkpoint_path.endswith('.pt'):
        #         self.generator = torch.load(f).cuda()
        #     else:
        #         self.generator = pickle.load(f)['G_ema'].cuda()
        self.generator = load_eg3d()

    def get_all_layers(self):
        layers = []
        # for child in self.generator.synthesis.children():
        for child in self.generator.backbone.children():
            layers += list(child.children())
            # return list(self.generator.synthesis.children())
        return layers

    def trainable_params(self):
        params = []
        for layer in self.get_training_layers():
            params.extend(layer.parameters())

        return params

    def get_training_layers(self, phase=None):
        # return self.get_all_layers()[:20] + self.get_all_layers()[23:] # freeze nerf
        return self.get_all_layers()

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.generator.children())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''

        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)
                
                # if hasattr(layer, "affine"):
                #     requires_grad(layer.affine, False)
                # else:
                #     for child_layer in layer.children():
                #         if hasattr(child_layer, "affine"):
                #             requires_grad(child_layer.affine, False)
                # if hasattr(layer, "torgb"):
                #     requires_grad(layer.torgb, False)

    def style(self, z_codes, c, truncation=0.7):
        if isinstance(z_codes, list):
            z_codes = z_codes[0]
        # print(z_codes.shape, c.shape)
        return self.generator.mapping(z_codes, c.repeat(z_codes.shape[0], 1), truncation_psi=truncation, truncation_cutoff=None)
        # return self.generator.mapping(z_codes[0], c.repeat(z_codes[0].shape[0], 1), truncation_psi=truncation, truncation_cutoff=None)

    def forward(self, styles, c=None, truncation=None, randomize_noise=True): # unused args for compatibility with SG2 interface
        noise_mode = 'random' if randomize_noise else 'const'
        return self.generator.synthesis(styles, c, noise_mode=noise_mode, force_fp32=True)['image'], None



class ZSSGAN(torch.nn.Module):
    def __init__(self, args):
        super(ZSSGAN, self).__init__()

        self.args = args

        self.device = 'cuda:0'

        # Set up twin generators
        self.generator_frozen = EG3DGenerator(args.frozen_gen_ckpt)
        self.generator_trainable = EG3DGenerator(args.train_gen_ckpt)
        
        # freeze relevant layers
        self.generator_frozen.freeze_layers()
        self.generator_frozen.eval()
        
        self.generator_trainable.freeze_layers()
        self.generator_trainable.unfreeze_layers(self.generator_trainable.get_training_layers(args.phase))
        self.generator_trainable.train()
        
        # self.c_front = torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,2.7, 0,0,0,1, 4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().to(self.device).reshape(1, -1)
        self.c_front = cal_canonical_c(device=self.device)

        # Losses
        self.clip_loss_models = {model_name: CLIPLoss(self.device, 
                                                      lambda_direction=args.lambda_direction, 
                                                      lambda_patch=args.lambda_patch, 
                                                      lambda_global=args.lambda_global, 
                                                      lambda_manifold=args.lambda_manifold, 
                                                      lambda_texture=args.lambda_texture,
                                                      clip_model=model_name) 
                                for model_name in args.clip_models}

        self.clip_model_weights = {model_name: weight for model_name, weight in zip(args.clip_models, args.clip_model_weights)}

        self.mse_loss  = torch.nn.MSELoss()

        self.source_class = args.source_class
        self.target_class = args.target_class

        self.auto_layer_k     = args.auto_layer_k
        self.auto_layer_iters = args.auto_layer_iters
        
        if args.target_img_list is not None:
            self.set_img2img_direction()

    def set_img2img_direction(self):
        with torch.no_grad():
            z_dim    = 64 if self.args.sgxl else 512
            sample_z = torch.randn(self.args.img2img_batch, z_dim, device=self.device)

            if self.args.sg3 or self.args.sgxl:
                generated = self.generator_trainable(self.generator_frozen.style([sample_z], self.c_front))[0]
            else:
                generated = self.generator_trainable([sample_z], self.c_front)[0]

            for _, model in self.clip_loss_models.items():
                direction = model.compute_img2img_direction(generated, self.args.target_img_list)

                model.target_direction = direction

    def determine_opt_layers(self):
        z_dim    = 64 if self.args.sgxl else 512
        sample_z = torch.randn(self.args.auto_layer_batch, z_dim, device=self.device)

        initial_w_codes = self.generator_frozen.style([sample_z], self.c_front)
        # initial_w_codes = initial_w_codes[0].unsqueeze(1).repeat(1, self.generator_frozen.generator.num_ws, 1)

        w_codes = torch.Tensor(initial_w_codes.cpu().detach().numpy()).to(self.device)

        w_codes.requires_grad = True

        w_optim = torch.optim.SGD([w_codes], lr=0.01)

        for _ in range(self.auto_layer_iters):
            # w_codes_for_gen = w_codes.unsqueeze(0)
            w_codes_for_gen = w_codes
            generated_from_w = self.generator_trainable(w_codes_for_gen, self.c_front.repeat(w_codes_for_gen.shape[0], 1))[0]

            w_loss = [self.clip_model_weights[model_name] * self.clip_loss_models[model_name].global_clip_loss(generated_from_w, self.target_class) 
                            for model_name in self.clip_model_weights.keys()]
            w_loss = torch.sum(torch.stack(w_loss))
            
            w_optim.zero_grad()
            w_loss.backward()
            w_optim.step()
        
        # layer_weights = torch.abs(w_codes - initial_w_codes).mean(dim=-1).mean(dim=0)
        # chosen_layer_idx = torch.topk(layer_weights, self.auto_layer_k)[1].cpu().numpy()

        all_layers = list(self.generator_trainable.get_all_layers())
        conv_inds = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 23, 24, 26, 27]
        rgb_inds = [1, 4, 7, 10, 13, 16, 19, 25, 28]
        conv_layers = []
        rgb_layers = []
        nerf_layers = []
        for i, layer in enumerate(all_layers):
            if i in conv_inds:
                conv_layers.append(layer)
            elif i in rgb_inds:
                rgb_layers.append(layer)
            else:
                nerf_layers.append(layer)

        # idx_to_layer = conv_layers + [rgb_layers[-1]]
        # chosen_layers = [idx_to_layer[idx] for idx in chosen_layer_idx] 
        chosen_layers = conv_layers

        # uncomment to add RGB layers to optimization.

        # for idx in chosen_layer_idx:
        #     if idx % 2 == 1 and idx >= 3 and idx < 14:
        #         chosen_layers.append(rgb_layers[(idx - 3) // 2])

        # uncomment to add learned constant to optimization
        # chosen_layers.append(all_layers[1])
                
        return chosen_layers

    def forward(
        self,
        styles,
        labels=None,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        c = labels if labels is not None else self.c_front
        if self.training and self.auto_layer_iters > 0:
            self.generator_trainable.unfreeze_layers()
            train_layers = self.determine_opt_layers()

            if not isinstance(train_layers, list):
                train_layers = [train_layers]

            self.generator_trainable.freeze_layers()
            self.generator_trainable.unfreeze_layers(train_layers)

        with torch.no_grad():
            if input_is_latent:
                w_styles = styles
            else:
                w_styles = self.generator_frozen.style(styles, c)
            
            frozen_img = self.generator_frozen(w_styles, c=c.repeat(w_styles.shape[0], 1), truncation=truncation, randomize_noise=randomize_noise)[0]

            if self.args.sg3 or self.args.sgxl:
                frozen_img = frozen_img + torch.randn_like(frozen_img) * 5e-4 # add random noise to add stochasticity in place of noise injections

        trainable_img = self.generator_trainable(w_styles, c=c.repeat(w_styles.shape[0], 1), truncation=truncation, randomize_noise=randomize_noise)[0]
        
        clip_loss = torch.sum(torch.stack([self.clip_model_weights[model_name] * self.clip_loss_models[model_name](frozen_img, self.source_class, trainable_img, self.target_class) for model_name in self.clip_model_weights.keys()]))

        return [frozen_img, trainable_img], clip_loss

    def pivot(self):
        par_frozen = dict(self.generator_frozen.named_parameters())
        par_train  = dict(self.generator_trainable.named_parameters())

        for k in par_frozen.keys():
            par_frozen[k] = par_train[k]
