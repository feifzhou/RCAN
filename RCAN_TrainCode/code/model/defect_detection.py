from model import common
from model.common import NNop
from model.rcan import *

import torch.nn as nn

def make_model(args, parent=False):
    return defect_detection(args)

class resnet(nn.Module):
    def __init__(self, num_hidden, kernel, configs):
        print('debug init resnet')
        super(resnet, self).__init__()
        # assert configs.kernel_size == 3:
        dim = configs.dim
        m = [default_conv(configs.n_colors, num_hidden[0], kernel, periodic=configs.periodic, dim=dim)]
        for i, nhid in enumerate(num_hidden):
            print('debug init resnet', i, nhid)
            m.append(resblock(nhid, kernel, bn=True, act=nn.ReLU(True), 
              dim=dim, periodic=configs.periodic))

        self.body = torch.nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)


class defect_detection(nn.Module):
    def __init__(self, args):
        super(defect_detection, self).__init__()
        conv=common.default_conv
        dim = args.dim
        self.dim = args.dim
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        act = nn.ReLU(True)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size, dim=dim)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks, dim=dim) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size, dim=dim))

        # define tail module
        if args.n_colors_out == -1:
            args.n_colors_out = args.n_colors
        modules_tail = [
            conv(n_feats, args.n_colors_out, kernel_size, dim=dim)]
        modules_tail+= [nn.Sigmoid()]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
