import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import cv2


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        in_nc = config.MODEL.IN_CHANNEL
        out_nc = config.MODEL.OUT_CHANNEL
        nf = config.MODEL.N_FEATURE
        nb = config.MODEL.N_BLOCK
        gc = config.MODEL.GROWTH_CHANNEL

        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
    def load_weight(self,model_path):
        loaded_model = torch.load(model_path)
        self.load_state_dict(loaded_model, strict=False)


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    # 4D: grid (B, C, H, W), 3D: (C, H, W), 2D: (H, W)
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), padding=0, normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)


if __name__ == '__main__':
    from config import config

    net = Network(config)

    sys.exit()

    # input and output paths
    input_folder_path = './LR'
    save_path = './results'

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')

    net = RRDBNet(config).to(device)
    print("model have {:.3f}M paramerters in total".format(sum(x.numel() for x in net.parameters())/1000000.0))
    model_path = './beby_gan.pth'
    net.load_weight(model_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = sorted(os.listdir(input_folder_path))

    for idx, f in enumerate(files) :
        if '.png' not in f:
            continue
        print(f)

        fpath = os.path.join(input_folder_path, f)
        img = cv2.imread(fpath)
        img = np.transpose(img[:, :, ::-1], (2, 0, 1)).astype(np.float32) / 255.0
        inp = torch.from_numpy(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out_hr = net(inp.cuda())

        output = tensor2img(out_hr)
        save_img_path = os.path.join(save_path, f)
        print('save_img_path', save_img_path)
        cv2.imwrite(save_img_path, output)

