"""
Partially modified from https://github.com/TengdaHan/MemDPC/blob/master/memdpc/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompassModel(nn.Module):
    def __init__(self, args):
        super(CompassModel, self).__init__()

        self.args = args
        from .select_backbone import select_resnet
        self.encoder, _, _, _, param = select_resnet('resnet18')

        if args.linear_prob:
            self.pred = nn.Sequential(
                nn.Linear(param['feature_size'], 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 4)
            )
        else:
            self.pred = nn.Conv2d(param['feature_size'], param['feature_size'], kernel_size=1, padding=0)
 
        self._initialize_weights(self.pred)
        self.load_pretrained_encoder_weights(args.pretrained_encoder_path)
    
    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 0.1)

    def load_pretrained_encoder_weights(self, pretrained_path):
        if pretrained_path:
            print('8888888888888888888888888888888')
            ckpt = torch.load(pretrained_path)['state_dict']  # COMPASS checkpoint format.
            ckpt2 = {}
            for key in ckpt:
                if key.startswith('backbone_rgb'):
                    ckpt2[key.replace('backbone_rgb.', '')] = ckpt[key]
                elif key.startswith('module.backbone'):
                    ckpt2[key.replace('module.backbone.', '')] = ckpt[key]
            self.encoder.load_state_dict(ckpt2)
            print('Successfully loaded pretrained checkpoint: {}.'.format(pretrained_path))
        else:
            print('Train from scratch.')
    
    def forward(self, x):
        # x: B, C, SL, H, W
        #x = x.unsqueeze(2)           # Shape: [B,C,H,W] -> [B,C,1,H,W].
        x = self.encoder(x)          # Shape: [B,C,1,H,W] -> [B,C',1,H',W']. FIXME: Need to check the shape of output here.

        if self.args.linear_prob:
            x = x.mean(dim=(2, 3, 4))    # Shape: [B,C',1,H',W'] -> [B,C'].
            x = self.pred(x)             # Shape: [B,C'] -> [B,C''].
            
        else:
            #TODO
            print('using convd')
            B, N, T, H, W = x.shape
            x = x.view(B, T, N, H, W)
            x = x.view(B*T, N, H, W)
            x = self.pred(x) 
            x = x.mean(dim=(1, 2, 3))
        return x

class CompassModel_K400(nn.Module):
    def __init__(self, args):
        super(CompassModel_K400, self).__init__()

        self.args = args
        #from .select_backbone import select_resnet
        #self.encoder, _, _, _, param = select_resnet('resnet18')

        from.select_backbone_memdpc import select_resnet
        self.encoder, param = select_resnet('resnet18')

        if args.linear_prob:
            self.pred = nn.Sequential(
                nn.Linear(param['feature_size'], 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 4)
            )
        else:
            self.pred = nn.Conv2d(param['feature_size'], param['feature_size'], kernel_size=1, padding=0)
 
        self._initialize_weights(self.pred)
        self.load_pretrained_encoder_weights(args.pretrained_encoder_path)
    
    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 0.1)

    def load_pretrained_encoder_weights(self, pretrained_path):
        if pretrained_path:
            ckpt = torch.load(pretrained_path)['state_dict']  # COMPASS checkpoint format.
            ckpt2 = {}
            for key in ckpt:
                print('-', key)
                if key.startswith('backbone_rgb'):
                    ckpt2[key.replace('backbone_rgb.', '')] = ckpt[key]
                elif key.startswith('module.backbone'):
                    ckpt2[key.replace('module.backbone.', '')] = ckpt[key]
            self.encoder.load_state_dict(ckpt2, strict=False)
            print('Successfully loaded pretrained checkpoint: {}.'.format(pretrained_path))
        else:
            print('Train from scratch.')
    
    def forward(self, x):
        # x: B, C, SL, H, W
        #x = x.unsqueeze(2)           # Shape: [B,C,H,W] -> [B,C,1,H,W].
        x = self.encoder(x)          # Shape: [B,C,1,H,W] -> [B,C',1,H',W']. FIXME: Need to check the shape of output here.

        if self.args.linear_prob:
            x = x.mean(dim=(2, 3, 4))    # Shape: [B,C',1,H',W'] -> [B,C'].
            x = self.pred(x)             # Shape: [B,C'] -> [B,C''].
            
        else:
            #TODO
            print('using convd')
            B, N, T, H, W = x.shape
            x = x.view(B, T, N, H, W)
            x = x.view(B*T, N, H, W)
            x = self.pred(x) 
            x = x.mean(dim=(1, 2, 3))
        return x

class CompassModel_GRU(nn.Module):
    def __init__(self, args):
        super(CompassModel_GRU, self).__init__()

        self.args = args
        from .select_backbone import select_resnet
        from .convrnn import ConvGRU
        self.encoder, _, _, _, param = select_resnet('resnet18')

        self.agg_f = ConvGRU(input_size=256,
                           hidden_size=256,
                           kernel_size=1,
                           num_layers=1)
        self.agg_b = ConvGRU(input_size=256,
                           hidden_size=256,
                           kernel_size=1,
                           num_layers=1)

        if args.linear_prob:
            self.pred = nn.Sequential(
                nn.Linear(param['feature_size']*2, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 4)
            )
        else:
            self.pred = nn.Conv2d(param['feature_size'], param['feature_size'], kernel_size=1, padding=0)
 
        self._initialize_weights(self.pred)
        self.load_pretrained_encoder_weights(args.pretrained_encoder_path)
    
    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 0.1)

    def load_pretrained_encoder_weights(self, pretrained_path):
        if pretrained_path:
            ckpt = torch.load(pretrained_path)['state_dict']  # COMPASS checkpoint format.
            ckpt2 = {}
            for key in ckpt:
                if key.startswith('backbone_rgb'):
                    ckpt2[key.replace('backbone_rgb.', '')] = ckpt[key]
                if key.startswith('GRU'):
                    ckpt2[key.replace('GRU', '')] = ckpt[key]
            self.encoder.load_state_dict(ckpt2)
            self.agg_f.load_state_dict(ckpt2, strict = False)
            self.agg_b.load_state_dict(ckpt2, strict = False)
            print('Successfully loaded pretrained checkpoint of encoder and GRU: {}.'.format(pretrained_path))
        else:
            print('Train from scratch.')
    
    def forward(self, x):
        # x: B, C, SL, H, W
        #x = x.unsqueeze(2)           # Shape: [B,C,H,W] -> [B,C,1,H,W].
        
        x = self.encoder(x)          # Shape: [B,C,1,H,W] -> [B,C',1,H',W']. FIXME: Need to check the shape of output here.

        # ==========
        B, C, SL, H, W = x.shape

        feature = F.relu(x)
        feature = F.avg_pool3d(feature, (1, 1, 1), stride=1)
        feature = feature.view(B, 1, C, H, W) # [B*N,D,last_size,last_size]
        
        context_forward, _ = self.agg_f(feature)
        context_forward = context_forward[:,-1,:].unsqueeze(1)
        context_forward = F.avg_pool3d(context_forward, (1, 7, 7), stride=1).squeeze(-1).squeeze(-1)

        feature_back = torch.flip(feature, dims=(1,))
        context_back, _ = self.agg_b(feature_back)
        context_back = context_back[:,-1,:].unsqueeze(1)
        context_back = F.avg_pool3d(context_back, (1, 7, 7), stride=1).squeeze(-1).squeeze(-1)

        context = torch.cat([context_forward, context_back], dim=-1) # B,N,C=2C

        x = context
        #===============

        x = x.view(B, 2*C)    # Shape: [B,C',1,H',W'] -> [B,C'].
        x = self.pred(x)             # Shape: [B,C'] -> [B,C''].
            
        return x

class CompassModelMem(nn.Module):
    def __init__(self, args):
        super(CompassModelMem, self).__init__()

        self.args = args
        from .select_backbone import select_resnet
        self.encoder, _, _, _, param = select_resnet('resnet18')

        self.last_size = 7
        if args.use_memory:
            self.mb = torch.nn.Parameter(torch.randn(1024, param['feature_size'], self.last_size, self.last_size))
            #self.mb = torch.nn.Parameter(
            #                    torch.randn(self.param['membanks_size'], self.param['feature_size']) / math.sqrt(self.param['feature_size'])
            #                    )
            self.key = nn.Conv2d(param['feature_size'], param['feature_size'], kernel_size=1, padding=0) #   
            self.key_ln = nn.LayerNorm(param['feature_size'])

            self.query = nn.Conv2d(param['feature_size'], param['feature_size'], kernel_size=1, padding=0) #  
            self.query_ln = nn.LayerNorm(param['feature_size'])

            

        if args.linear_prob:
            self.pred = nn.Sequential(
                nn.Linear(param['feature_size']*2, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 4)
            )
        else:
            self.pred = nn.Conv2d(param['feature_size'], param['feature_size'], kernel_size=1, padding=0)
 
        self._initialize_weights(self.pred)
        self.load_pretrained_encoder_weights(args.pretrained_encoder_path)
    
    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 0.1)

    def load_pretrained_encoder_weights(self, pretrained_path):
        if pretrained_path:
            ckpt = torch.load(pretrained_path)['state_dict']  # COMPASS checkpoint format.
            ckpt2 = {}
            for key in ckpt: 
                print('-', key)
                if key.startswith('backbone_rgb'):
                    ckpt2[key.replace('backbone_rgb.', '')] = ckpt[key]
                if key.startswith('mb'):
                    ckpt2[key.replace('mb.', '')] = ckpt[key]
                #if key.startswith('network_key'):
                #    ckpt2[key.replace('network_key.', '')] = ckpt[key]
                #if key.startswith('network_query_rgb'):
                #    ckpt2[key.replace('network_query_rgb.', '')] = ckpt[key]
                
            for key in ckpt2:
                print('--ckpt2', key) 
            import pdb
            pdb.set_trace()
            self.mb.load_state_dict(ckpt2)
            self.encoder.load_state_dict(ckpt2)
            #self.key.load_state_dict(ckpt2)
            #self.qeury.load_state_dict(ckpt2)
            

            print('Successfully loaded pretrained checkpoint of encoder and GRU: {}.'.format(pretrained_path))
        else:
            print('Train from scratch.')
    
    def forward(self, x):
        # x: B, C, SL, H, W
        #x = x.unsqueeze(2)           # Shape: [B,C,H,W] -> [B,C,1,H,W].
        
        x = self.encoder(x)          # Shape: [B,C,1,H,W] -> [B,C',1,H',W']. FIXME: Need to check the shape of output here.

        # ==========
        B, C, SL, H, W = x.shape

        feature = F.relu(x)
        feature = F.avg_pool3d(feature, (1, 1, 1), stride=1)
        feature = feature.view(B, 1, C, H, W) # [B*N,D,last_size,last_size]
        
        #context_forward, _ = self.agg_f(feature)
        #context_forward = context_forward[:,-1,:].unsqueeze(1)
        #context_forward = F.avg_pool3d(context_forward, (1, 7, 7), stride=1).squeeze(-1).squeeze(-1)

        #feature_back = torch.flip(feature, dims=(1,))
        #context_back, _ = self.agg_b(feature_back)
        #context_back = context_back[:,-1,:].unsqueeze(1)
        #context_back = F.avg_pool3d(context_back, (1, 7, 7), stride=1).squeeze(-1).squeeze(-1)

        #context = torch.cat([context_forward, context_back], dim=-1) # B,N,C=2C

        #x = context
        #===============

        x = x.view(B, C)    # Shape: [B,C',1,H',W'] -> [B,C'].
        x = self.pred(x)             # Shape: [B,C'] -> [B,C''].
            
        return x