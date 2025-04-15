import torch
import torch.nn as nn
from torch.nn import init, Parameter
import functools
from torch.optim import lr_scheduler
import torch
from .feature_decoupling import *
from .cbam import *
from .fusion_transformer import *
class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=4, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

from torchstat import stat
from thop import profile
from thop import clever_format
def define_MHUG(n_input_modal, input_nc, output_nc, ngf, norm='batch', use_dropout=False,
               init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Create a multi-head(multi-branch) generator
    """
    norm_layer = get_norm_layer(norm_type=norm)
    net = MultiHeadUnetGenerator(n_input_modal, input_nc, output_nc, ngf, norm_layer=norm_layer,
                                   use_dropout=use_dropout)

    return init_net(net, init_type, init_gain, gpu_ids)


class SingleModalAttetionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SingleModalAttetionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + x * y.expand_as(x)

class MultiHeadUnetGenerator(nn.Module):
    """Resnet-based Multi-Head Generator"""

    def __init__(self, n_input_modal, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            n_input_modal(int)  -- the number of input modal
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        super(MultiHeadUnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.n_input_modal = n_input_modal
        n_downsampling = 2
        self.n_downsampling = n_downsampling

        for down_idx in range(3):
            down = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                     norm_layer(ngf),
                     nn.ReLU(True)]
            self.add_module('encoder_{}_0'.format(down_idx), nn.Sequential(*down))

            for i in range(n_downsampling):
                mult = 2 ** i
                down = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                         norm_layer(ngf * mult * 2),
                         nn.ReLU(True)]

                self.add_module('encoder_{}_{}'.format(down_idx, i + 1), nn.Sequential(*down))
        self.baseFeature =MambaLayer(dim=256)
        # self.baseFeature = BaseFeatureExtraction(dim=256, num_heads=8)
        self.detailFeature = DetailFeatureExtraction()
        self.dim_reduce_conv = nn.Conv2d(ngf * 8 * 2, ngf * 8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.dim_reduce_conv_3_3 = nn.Conv2d(ngf * 8 * 3, ngf * 8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.dim_reduce_conv_3_2 = nn.Conv2d(ngf * 4 * 3, ngf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.dim_reduce_conv_3_1 = nn.Conv2d(ngf * 2 * 3, ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.dim_reduce_conv_3_0 = nn.Conv2d(ngf * 1 * 3, ngf * 1, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.dim_reduce_conv_3 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),
                                               norm_layer(ngf * 4),
                                               nn.ReLU(True))
        self.dim_reduce_conv_2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),
                                               norm_layer(ngf * 2),
                                               nn.ReLU(True))
        self.dim_reduce_conv_1 = nn.Sequential(nn.ConvTranspose2d(ngf * 2 * 3, ngf * 1, kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),
                                               norm_layer(ngf * 1),
                                               nn.ReLU(True))
        self.dim_reduce_conv_0 = nn.Sequential(nn.ConvTranspose2d(ngf * 1 * 3, ngf , kernel_size=1),
                                               norm_layer(ngf),
                                               nn.ReLU(True))
        self.dim_reduce_3 = nn.Conv2d(ngf * 8 * 2, ngf * 8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.dim_reduce_2 = nn.Conv2d(ngf * 4 * 2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.dim_reduce_1 = nn.Conv2d(ngf * 2 * 2, ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.dim_reduce_0 = nn.Conv2d(ngf * 1 * 2, ngf * 1, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.cbam_3 = CBAM(ngf * 8 * 3)
        self.cbam_2 = CBAM(ngf * 4 * 3)
        self.cbam_1 = CBAM(ngf * 2 * 3)
        self.cbam_0 = CBAM(ngf * 1 * 3)

        self.self_attention_3 = SingleModalAttetionLayer(channel=ngf * 8 * 2)
        self.self_attention_2 = SingleModalAttetionLayer(channel=ngf * 4 * 2)
        self.self_attention_1 = SingleModalAttetionLayer(channel=ngf * 2 * 3)
        self.self_attention_0 = SingleModalAttetionLayer(channel=ngf * 1 * 3)
        self.tanh = nn.Tanh()

        for up_idx in range(2):   # segment and texture
            mult = 2 ** n_downsampling
            for i in range(n_downsampling, 0, -1):  # add upsampling layers
                mult = 2 ** i
                up = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                            norm_layer(int(ngf * mult / 2)),
                            nn.ReLU(True)]
                self.add_module('decoder_{}_{}'.format(up_idx, i), nn.Sequential(*up))
            if up_idx == 1:  # b
                up = [nn.ReflectionPad2d(3),
                      nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                      nn.Sigmoid()]
            else:  # d
                up = [nn.ReflectionPad2d(3),
                      nn.Conv2d(ngf, output_nc * 4, kernel_size=7, padding=0),
                      nn.ReLU(True),
                      nn.Softmax(dim=1)]
            self.add_module('decoder_{}_0'.format(up_idx), nn.Sequential(*up))
        self.out = nn.Sequential(
                                 nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())

    def forward(self, input, train=False):
        """Standard forward"""
        input = input.cuda()
        inputs = torch.chunk(input, self.n_input_modal, dim=1)
        encoder_features = []

        x_0_0 = self.__getattr__('encoder_{}_{}'.format(0, 0))(inputs[0])  # 1 -> 64
        x_0_1 = self.__getattr__('encoder_{}_{}'.format(0, 1))(x_0_0)      # 64 -> 128
        x_0_2 = self.__getattr__('encoder_{}_{}'.format(0, 2))(x_0_1)      # 128 -> 256
        # x_0_3 = self.__getattr__('encoder_{}_{}'.format(0, 3))(x_0_2)      # 256 -> 512
        b_0 = self.baseFeature(x_0_2)
        d_0 = self.detailFeature(x_0_2)

        x_1_0 = self.__getattr__('encoder_{}_{}'.format(1, 0))(inputs[1])
        x_1_1 = self.__getattr__('encoder_{}_{}'.format(1, 1))(x_1_0)
        x_1_2 = self.__getattr__('encoder_{}_{}'.format(1, 2))(x_1_1)
        # x_1_3 = self.__getattr__('encoder_{}_{}'.format(1, 3))(x_1_2)
        b_1 = self.baseFeature(x_1_2)
        d_1 = self.detailFeature(x_1_2)

        x_2_0 = self.__getattr__('encoder_{}_{}'.format(2, 0))(inputs[2])
        x_2_1 = self.__getattr__('encoder_{}_{}'.format(2, 1))(x_2_0)
        x_2_2 = self.__getattr__('encoder_{}_{}'.format(2, 2))(x_2_1)
        # x_2_3 = self.__getattr__('encoder_{}_{}'.format(2, 3))(x_2_2)
        b_2 = self.baseFeature(x_2_2)     #
        d_2 = self.detailFeature(x_2_2)   #

        b = b_0 + b_1 + b_2
        d = d_0 + d_1 + d_2
        
        s_0 = self.dim_reduce_conv_3_0(self.cbam_0(torch.cat((x_0_0, x_1_0, x_2_0), dim=1)))
        s_1 = self.dim_reduce_conv_3_1(self.cbam_1(torch.cat((x_0_1, x_1_1, x_2_1), dim=1)))
        s_2 = self.dim_reduce_conv_3_2(self.cbam_2(torch.cat((x_0_2, x_1_2, x_2_2), dim=1)))
        # s_3 = self.dim_reduce_conv_3_3(self.cbam_3(torch.cat((x_0_3, x_1_3, x_2_3), dim=1)))

        # y_b_3 = self.__getattr__('decoder_{}_{}'.format(0, 3))(b)         # 512 -> 256   纹理
        # y_b_3 = self.dim_reduce_2(torch.cat((y_b_3, s_2), dim=1))
        y_b_2 = self.__getattr__('decoder_{}_{}'.format(0, 2))(b)     # 256 -> 128
        y_b_2 = self.dim_reduce_1(torch.cat((y_b_2, s_1), dim=1))
        y_b_1 = self.__getattr__('decoder_{}_{}'.format(0, 1))(y_b_2)     # 128 -> 64
        y_b_1 = self.dim_reduce_0(torch.cat((y_b_1, s_0), dim=1))
        y_b_0 = self.__getattr__('decoder_{}_{}'.format(0, 0))(y_b_1)     #  64 -> 1


        
        # y_d_3 = self.__getattr__('decoder_{}_{}'.format(1, 3))(d)     # 分割
        # y_d_3 = self.dim_reduce_2(torch.cat((y_d_3, s_2), dim=1))
        y_d_2 = self.__getattr__('decoder_{}_{}'.format(1, 2))(d)
        y_d_2 = self.dim_reduce_1(torch.cat((y_d_2, s_1), dim=1))
        y_d_1 = self.__getattr__('decoder_{}_{}'.format(1, 1))(y_d_2)
        y_d_1 = self.dim_reduce_0(torch.cat((y_d_1, s_0), dim=1))
        y_d_0 = self.__getattr__('decoder_{}_{}'.format(1, 0))(y_d_1)



        # u = self.dim_reduce_conv(torch.cat((b, d), dim=1))
        u_2= self.dim_reduce_conv_2(self.self_attention_2(torch.cat((b, d), dim=1)))
        # u_2 = self.dim_reduce_conv_2(self.self_attention_2(torch.cat((y_b_3, y_d_3, u_3), dim=1)))
        u_1 = self.dim_reduce_conv_1(self.self_attention_1(torch.cat((y_b_2, y_d_2, u_2), dim=1)))
        u_0 = self.dim_reduce_conv_0(self.self_attention_0(torch.cat((y_b_1, y_d_1, u_1), dim=1)))
        
  
        # out = u_0
        # print(y_d_0.shape)
        # y_b = torch.argmax(y_b_0, 1).unsqueeze(1)
        # y_b = y_b / 4.0 * 255.0
        # print(y_b_0)
        out = self.out(u_0)

        if train:
            return out, y_b_0, y_d_0, b_0, b_1, b_2, d_0, d_1, d_2
        else:
            return out, y_b_0, y_d_0, b_0, b_1, b_2, d_0, d_1, d_2


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
