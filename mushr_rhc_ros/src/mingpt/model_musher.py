"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

from base64 import encode
import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np

import sys
sys.path.append('../')
from models.compass.select_backbone import select_resnet

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, max_timestep, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.max_timestep = max_timestep
        for k,v in kwargs.items():
            setattr(self, k, v)
            print(k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.device = device

        self.model_type = config.model_type
        self.use_pred_state = config.use_pred_state

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        # self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        action_tanh=True

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))


        if config.state_tokenizer == 'conv2D':
            self.state_encoder = nn.Sequential(nn.Conv2d(1, 32, 8, stride=4, padding=0), nn.ReLU(),
                                     nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                     nn.Flatten(), nn.Linear(36864, config.n_embd), nn.Tanh())
        elif config.state_tokenizer == 'compass':
            #from select_backbone import select_resnet
            self.state_encoder, _, _, _, param = select_resnet('resnet18')
            self.load_pretrained_encoder_weights(config.pretrained_encoder_path)
            self.linear = nn.Linear(256, 128)
            self._initialize_weights(self.linear)
        
        elif config.state_tokenizer == 'FCL':
            self.state_encoder = nn.Sequential(
                                 nn.Linear(40000, 2048),
                                 nn.Tanh(),
                                 nn.Linear(2048, 512),
                                 nn.Tanh(),
                                 nn.Linear(512, config.n_embd))

        # self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        if config.train_mode == 'map': 
            # add map decoder
            encoded_feat_dim = (config.n_embd * config.block_size) // 2
            #self.map_decoder = MapDecoder_4x_Deconv(encoded_feat_dim)
            #self.map_decoder = nn.Sequential(nn.Linear(encoded_feat_dim, 1024), nn.Tanh(),
            #                                 nn.Linear(1024, 2048), nn.Tanh(),
            #                                 nn.Linear(2048, 64*64), nn.Sigmoid())
            # MLP map decoder
            #self.map_decoder = nn.Sequential(nn.Linear(encoded_feat_dim, 1024), nn.Tanh(),
            #                                 nn.Linear(1024, 2048), nn.Tanh(),
            #                                 nn.Linear(2048, 64*64), nn.Tanh())

            # conv2d map decoder
            self.map_decoder = nn.Sequential(nn.Linear(encoded_feat_dim, 4096), nn.Tanh(),
                                             Reshape(16, 16, 16),
                                             MapDecoder_2x_Deconv(16))


        if config.loss == 'cross_entropy':
            self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh()) 
            nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)
            
        elif config.loss == 'MSE':
            # ======== linear action space ===========================
            #self.action_embeddings = nn.Linear(1, config.n_embd)
            #self.action_embeddings = nn.Sequential(nn.Linear(4, config.n_embd), nn.Tanh())
            #self._initialize_weights(self.action_embeddings)
            self.action_embeddings = nn.Sequential(
                                     nn.Linear(1, 32),
                                     nn.Tanh(),
                                     nn.Linear(32, 64),
                                     nn.Tanh(),
                                     nn.Linear(64, config.n_embd)
            )

        self.predict_action = nn.Sequential(
                              nn.Linear(config.n_embd, 64),
                              nn.Tanh(),
                              nn.Linear(64, 32),
                              nn.Tanh(),
                              nn.Linear(32, 1)
        )

        self.predict_state = nn.Sequential(
            *([nn.Linear(config.n_embd, config.n_embd)] + ([nn.Tanh()] if action_tanh else []))
        )

        criterion = torch.nn.MSELoss(reduction='mean')
        self.criterion = criterion.cuda(device)

        # ===== linear projection for MSE loss on continous action ====

        ##=====debug position embeddings==========
        #self.position_embeddings = nn.Parameter(torch.zeros(2, 8*2 , self.config.n_embd))

        #self.device = torch.cuda.current_device()

    def reconstruction_loss(self, pred, target):
        loss = F.l1_loss(pred, target)
        return loss

    def load_pretrained_encoder_weights(self, pretrained_path):
        if pretrained_path:
            ckpt = torch.load(pretrained_path)['state_dict']  # COMPASS checkpoint format.
            ckpt2 = {}
            for key in ckpt:
                if key.startswith('backbone_rgb'):
                    ckpt2[key.replace('backbone_rgb.', '')] = ckpt[key]
                elif key.startswith('module.backbone'):
                    ckpt2[key.replace('module.backbone.', '')] = ckpt[key]
            self.state_encoder.load_state_dict(ckpt2)
            print('Successfully loaded pretrained checkpoint: {}.'.format(pretrained_path))
        else:
            print('Train from scratch.')

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 0.1)


    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm3d, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, and action
    def forward(self, states, actions, targets=None, timesteps=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # timesteps: (batch, 1, 1) 
        if self.config.state_tokenizer == 'conv2D':
            state_embeddings = self.state_encoder(states.reshape(-1, 1, 224, 224).type(torch.float32).contiguous()) # (batch * block_size, n_embd)
            state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd) # (batch, block_size, n_embd)
        elif self.config.state_tokenizer == 'FCL':
            state_embeddings = self.state_encoder(states.view(-1, 40000))
            state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd)
        elif self.config.state_tokenizer == 'compass':
            B, SL, _ = states.shape
            tmp = states.view(B*SL, -1).view(-1, 3, 224, 224)
            tmp = tmp.unsqueeze(2)
            tmp = self.state_encoder(tmp)
            tmp = tmp.mean(dim=(2,3,4))
            tmp = self.linear(tmp)
            state_embeddings = tmp.view(B, SL, -1)
            
        else:
            print('Not supported!')

        if actions is not None and self.model_type == 'GPT':
            if self.config.loss == 'MSE':
                B, N, C = actions.shape
                tmp = actions.view(B*N, C)
                action_embeddings = self.action_embeddings(tmp).view(B, N, -1) # (batch, block_size, n_embd)
            elif self.config.loss == 'cross_entropy':
                action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            
            #token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=self.device)

            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'GPT': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()
        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd

        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        #position_embeddings = torch.gather(all_global_pos_emb, 1, self.pos_emb[:, :token_embeddings.shape[1], :].type(torch.long))

        #TODO  debug position_embeddings
        #position_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=self.device)

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        if self.config.train_mode == 'e2e':
            if self.config.loss == 'cross_entropy':
                logits = self.head(x)
                logits = logits[:, ::2, :]
            elif self.config.loss == 'MSE':
                action_preds = self.predict_action(x[:, ::2, :])
        elif self.config.train_mode == 'map':
            percep_feat = x[:, ::2, :]
            B, N, D = percep_feat.shape
            feat = percep_feat.reshape(B, -1)  # reshape to a vector
            # reshape to B, C, H, W
            #feat = torch.unsqueeze(feat, dim=2)
            #feat = torch.unsqueeze(feat, dim=3)
            #feat = feat.repeat(1, 1, 25, 25) # replicate along H, W 

            map_recon = self.map_decoder(feat)
        else:
            print('Not support!')


        if self.use_pred_state:
            state_preds = self.predict_state(x[:, 1::2, :])

        #if actions is not None and self.model_type == 'GPT':
        #    logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        #elif actions is None and self.model_type == 'GPT':
        #    logits = logits # for completeness
        #else:
        #    raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            if self.config.train_mode == 'map':
                #gt_map = gt_map.reshape(-1, 64*64)
                #loss = self.reconstruction_loss(map_recon, gt_map)
                loss = self.criterion(map_recon.reshape(-1, 64, 64), gt_map)

                #map_recon_tmp = F.softmax(map_recon_tmp)
                #loss = nn.CrossEntropyLoss(map_recon_tmp, gt_map)
                #loss = F.binary_cross_entropy(map_recon, gt_map)

                # debugging
                #print('recon', map_recon, map_recon.max().item(), map_recon.min().item())
                #print('gt', gt_map, gt_map.max().item(), gt_map.min().item())
                #print('loss', loss) 
                #import matplotlib.pyplot as plt
                #gt_tmp = gt_map[0, :].reshape(64, 64)
                #map_tmp = map_recon[0, :].reshape(64, 64)
                #plt.imshow(gt_tmp.detach().cpu())
                #plt.savefig('gttmp.png')
                #plt.imshow(map_tmp.detach().cpu())
                #plt.savefig('maptmp.png')

                return map_recon, loss
            elif self.config.train_mode == 'e2e':
                if self.config.loss == 'cross_entropy':
                #targets = targets.clamp(min=0, max=50)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1).long())
                    return logits, loss
                elif self.config.loss == 'MSE':
                    # loss over N timesteps
                    loss = self.criterion(actions, torch.tanh(action_preds) )
                    return action_preds, loss
            if self.use_pred_state:
                loss_state = self.criterion(state_embeddings, state_preds)
                loss = loss + loss_state
        #return action_preds, loss

class MapDecoder_4x_Deconv(nn.Module):
    def __init__(self, in_channels=384):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvTranspose2d_FixOutputSize(nn.ConvTranspose2d(in_channels, 256, kernel_size=3, stride=2, padding=1), output_size=(50, 50)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ConvTranspose2d_FixOutputSize(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1), output_size=(100, 100)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ConvTranspose2d_FixOutputSize(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1), output_size=(200, 200)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ConvTranspose2d_FixOutputSize(nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1), output_size=(400, 400)),
        )

    def forward(self, x):
        return self.decoder(x)

class MapDecoder_2x_Deconv(nn.Module):
    def __init__(self, in_channels=768):
        super().__init__()

        # The parameters for ConvTranspose2D are from the PyTorch repo.
        # Ref: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        # Ref: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        # Ref: https://discuss.pytorch.org/t/the-output-size-of-convtranspose2d-differs-from-the-expected-output-size/1876/13
        # Ref: (padding) https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11
        self.decoder = nn.Sequential(
            ConvTranspose2d_FixOutputSize(nn.ConvTranspose2d(in_channels, 8, kernel_size=3, stride=2, padding=1), output_size=(32, 32)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            ConvTranspose2d_FixOutputSize(nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1), output_size=(64, 64)),
        )

    def forward(self, x):
        return self.decoder(x)

class ConvTranspose2d_FixOutputSize(nn.Module):
    """ 
    A wrapper to fix the output size of ConvTranspose2D.
    Ref: https://discuss.pytorch.org/t/the-output-size-of-convtranspose2d-differs-from-the-expected-output-size/1876/13
    Ref: (other alternatives) https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    def __init__(self, conv, output_size):
        super(ConvTranspose2d_FixOutputSize, self).__init__()
        self.output_size = output_size
        self.conv = conv
        
    def forward(self, x):
        x = self.conv(x, output_size=self.output_size)
        return x

class Reshape(nn.Module):
    def __init__(self, *args):
     super(Reshape, self).__init__()
     self.shape = args
    def forward(self, x):
     return x.view((x.size(0),)+self.shape)