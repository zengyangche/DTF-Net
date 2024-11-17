from collections import OrderedDict
import torch
from torch import nn
from .base_model import BaseModel
from . import networks
import numpy as np
import time
import os
from .ssim import *
from .dice_score import *
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES']="0"

class UnetAttentionModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.cnt = 0
        self.pretrain_cnt = 200
        self.n_input_modal = opt.n_input_modal
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>

        self.loss_names = ['G', 'G_GAN', 'G_L1', 'Seg', 'Tex', 'G_ssim', 'G_cos', 'D_real', 'D_fake', 'D']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_MHUG(self.n_input_modal, opt.input_nc, opt.output_nc, opt.ngf,
                                        opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(self.n_input_modal*(opt.input_nc) + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionHuber = torch.nn.HuberLoss(reduction='mean', delta=1.0).to(self.device)
            self.criterionCE = torch.nn.CrossEntropyLoss().to(self.device)
            self.criterionSSIM = SSIM(window_size=15).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.

            self.optimizer_G = torch.optim.RMSprop(self.netG.parameters(), lr=opt.lr)
            self.optimizer_D = torch.optim.RMSprop(self.netD.parameters(), lr=opt.lr)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.criterionL2 = torch.nn.MSELoss()
            # self.criterionKL = torch.nn.KLDivLoss()
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        if self.isTrain:
            self.real_S = input['S'].to(self.device)
        self.real_T = input['T'].to(self.device)
        self.modal_names = [i[0] for i in input['modal_names']]

    
    def plot_frequency_components(self, features, titles, save_folder):
        import numpy as np
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))  # 创建一个2行3列的子图
        for i, (feature, title) in enumerate(zip(features, titles)):
            ax = axs[i // 3, i % 3]  # 定位到具体的子图

            # 转换为频域：提取特征图的第一个批次和第一个通道
            spatial_feature = feature[0, 0].cpu().detach().numpy()
            # 快速傅里叶变换
            frequency_feature = np.fft.fftshift(np.fft.fft2(spatial_feature))
            # 计算幅度谱
            magnitude_spectrum = np.log(np.abs(frequency_feature) + 1)

            # 绘制频域图
            ax.imshow(magnitude_spectrum, cmap='Blues')
            ax.set_title(title)
            ax.axis('off')  # 关闭坐标轴
        # 保存图像，文件名按顺序命名
        save_path = os.path.join(save_folder, f"frequency_component_{self.cnt+1}.png")
        self.cnt = self.cnt + 1
        plt.savefig(save_path, dpi=300)
        plt.close(fig)  # 关闭当前图，节省内存


    def forward(self, train=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if train:
          
            self.fake_B, self.y_b_0, self.y_d_0, self.b_0, self.b_1, self.b_2, self.d_0, self.d_1, self.d_2 = self.netG(self.real_A, True)
        else:

            # stat(self.netG.module, (3, 256, 256))
            start_time = time.time()  # 记录开始时间
            self.fake_B, self.y_b_0, self.y_d_0, self.b_0, self.b_1, self.b_2, self.d_0, self.d_1, self.d_2= self.netG(self.real_A, train)  # G(A)
            end_time = time.time()  # 记录结束时间
            # 计算并打印每次推理时间
            inference_time = end_time - start_time
            print(f"Single sample inference time: {inference_time:.6f} seconds")
            # 在推理模式下测量时间
 
            
            features = [self.b_0, self.b_1, self.b_2, self.d_0, self.d_1, self.d_2]
            titles = ['low-mod_1', 'low-mod_2', 'low-mod_3', 'high-mod_1', 'high-mod_2', 'high-mod_3']

            # 调用函数，绘制并保存频域图
            # self.plot_frequency_components(features, titles, "keshihua")


#             import numpy as np
#             from sklearn.decomposition import PCA
#             import matplotlib.pyplot as plt
#             from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图模块
            
#             # 假设 self.d_0、self.d_1 和 self.d_2 是你的三个特征
#             d_0 = self.d_0.cpu().detach().numpy()
#             d_1 = self.d_1.cpu().detach().numpy()
#             d_2 = self.d_2.cpu().detach().numpy()
            
#             # 将一维数据转换为二维
#             d_0 = d_0.reshape((d_0.shape[1], -1))
#             d_1 = d_1.reshape((d_1.shape[1], -1))
#             d_2 = d_2.reshape((d_2.shape[1], -1))
            
#             # 使用 PCA 分别对三个特征进行降维
#             pca = PCA(n_components=3)  # 改为3D
#             d_0_pca = pca.fit_transform(d_0)
#             d_1_pca = pca.fit_transform(d_1)
#             d_2_pca = pca.fit_transform(d_2)
            
#             # 归一化
#             d_0_pca = (d_0_pca - np.min(d_0_pca)) / (np.max(d_0_pca) - np.min(d_0_pca))
#             d_1_pca = (d_1_pca - np.min(d_1_pca)) / (np.max(d_1_pca) - np.min(d_1_pca))
#             d_2_pca = (d_2_pca - np.min(d_2_pca)) / (np.max(d_2_pca) - np.min(d_2_pca))
            
#             # 可视化
#             fig = plt.figure(figsize=(8, 6))
#             ax = fig.add_subplot(111, projection='3d')  # 添加3D坐标轴
            
#             colors = ['r', 'g', 'b']  # 每个特征对应一个颜色
            
#             ax.scatter(d_0_pca[:, 0], d_0_pca[:, 1], d_0_pca[:, 2], color=colors[0], label='T1')
#             ax.scatter(d_1_pca[:, 0], d_1_pca[:, 1], d_1_pca[:, 2], color=colors[1], label='T2')
#             ax.scatter(d_2_pca[:, 0], d_2_pca[:, 1], d_2_pca[:, 2], color=colors[2], label='Flair')
            
#             # ax.set_title('Intra-Modal Specific Feature')
#             ax.set_xlabel('X')
#             ax.set_ylabel('Y')
#             ax.set_zlabel('Z')
#             ax.legend()
#             plt.show()
            
#             d_0 = self.b_0.cpu().detach().numpy()
#             d_1 = self.b_1.cpu().detach().numpy()
#             d_2 = self.b_2.cpu().detach().numpy()
            
#             # 将一维数据转换为二维
#             d_0 = d_0.reshape((d_0.shape[1], -1))
#             d_1 = d_1.reshape((d_1.shape[1], -1))
#             d_2 = d_2.reshape((d_2.shape[1], -1))
            
#             # 使用 PCA 分别对三个特征进行降维
#             pca = PCA(n_components=3)  # 改为3D
#             d_0_pca = pca.fit_transform(d_0)
#             d_1_pca = pca.fit_transform(d_1)
#             d_2_pca = pca.fit_transform(d_2)
            
#             # 归一化
#             d_0_pca = (d_0_pca - np.min(d_0_pca)) / (np.max(d_0_pca) - np.min(d_0_pca))
#             d_1_pca = (d_1_pca - np.min(d_1_pca)) / (np.max(d_1_pca) - np.min(d_1_pca))
#             d_2_pca = (d_2_pca - np.min(d_2_pca)) / (np.max(d_2_pca) - np.min(d_2_pca))
            
#             # 可视化
#             fig = plt.figure(figsize=(8, 6))
#             ax = fig.add_subplot(111, projection='3d')  # 添加3D坐标轴
            
#             colors = ['r', 'g', 'b']  # 每个特征对应一个颜色
            
#             ax.scatter(d_0_pca[:, 0], d_0_pca[:, 1], d_0_pca[:, 2], color=colors[0], label='T1')
#             ax.scatter(d_1_pca[:, 0], d_1_pca[:, 1], d_1_pca[:, 2], color=colors[1], label='T2')
#             ax.scatter(d_2_pca[:, 0], d_2_pca[:, 1], d_2_pca[:, 2], color=colors[2], label='Flair')
            
#             # ax.set_title('Inter-Modal Related Feature')
#             ax.set_xlabel('X')
#             ax.set_ylabel('Y')
#             ax.set_zlabel('Z')
#             ax.legend()
#             plt.show()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False) 
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB.detach())
        self.loss_D_real = self.criterionGAN(pred_real, True) 
        # combine loss and calculate gradients
        self.loss_D = self.loss_D_real  - self.loss_D_fake
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach()) 

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 100
        self.loss_G_ssim = (1 - self.criterionSSIM(self.fake_B, self.real_B)) * 100
        # print(self.y_d_0.max())
        # print(self.y_b_0)
        self.loss_Tex = self.criterionHuber(self.y_d_0, self.real_T)
        
        num_classes = 4  # 类别数量
        label_mapping = {0: 0, 1: 1, 2: 2, 4: 3}
        # label_mapping = {0: 0, 2: 1}
        mapped_labels = torch.zeros_like(self.real_S, dtype=torch.long)
        # print(self.real_S)

        # print(self.real_S)
        for label, mapped_label in label_mapping.items():
            mapped_labels[self.real_S == label] = mapped_label
        # print(mapped_labels.shape)
        # self.real_S = mapped_labels
        mask_one_hot = torch.nn.functional.one_hot(mapped_labels, num_classes)
        # print(self.y_d_0.shape)
        # print(mask_one_hot.shape)
        mask_one_hot = torch.squeeze(mask_one_hot,1)
        loss_seg_ce = F.nll_loss(self.y_b_0, torch.squeeze(mapped_labels, 1))
        self.loss_Seg = dice_loss(self.y_b_0.float(), mask_one_hot.permute(0, 3, 1, 2).float(), multiclass=True) +\
                        loss_seg_ce
        # uncorrelated
        self.loss_cor = F.cosine_similarity(self.d_0, self.d_1).mean() + F.cosine_similarity(self.d_1, self.d_2).mean() + F.cosine_similarity(self.d_0, self.d_2).mean()
        #  correlated
        self.loss_uncor =  (1 - F.cosine_similarity(self.b_0, self.b_1).mean()) + \
                           (1 - F.cosine_similarity(self.b_1, self.b_2).mean()) + \
                           (1 - F.cosine_similarity(self.b_2, self.b_0).mean())
        self.loss_G_cos =  self.loss_cor  + self.loss_uncor        # 2
        # combine loss and calculate gradients

        self.loss_G = self.loss_G_GAN  + self.loss_G_L1 + self.loss_G_ssim + \
                          self.loss_Tex + self.loss_Seg + self.loss_G_cos
      
        # self.cnt = self.cnt + 1
        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward(True)  # compute fake images: G(A)
        for p in self.netD.parameters():
            p.data.clamp_(-0.01, 0.01)

        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        self.cnt = self.cnt + 1

    def compute_visuals(self):
        """Calculate additional output images for tensorboard visualization"""
        pass

    def get_current_visuals(self):
        modal_imgs = []
        for i in range(self.n_input_modal):
            modal_imgs.append(torch.squeeze(self.real_A[:, i * (self.opt.input_nc):i * (self.opt.input_nc) + self.opt.input_nc, :, :],0))
        # 创建映射字典，将3映射为4
        self.y_b_0 = np.array(self.y_b_0.cpu().detach()).argmax(1)

        self.y_b_0 = torch.tensor(self.y_b_0)
        # label_mapping = {0: 0, 1: 2}
        label_mapping = {0: 0, 1: 1, 2: 2, 3: 4}

        mapped_labels = torch.zeros_like(self.y_b_0, dtype=torch.uint8)

        for label, mapped_label in label_mapping.items():
            mapped_labels[self.y_b_0 == label] = mapped_label
        # print(mapped_labels)
        # 现在，mapped_labels 包含了映射后的标签
        self.y_b_0 = mapped_labels

        self.y_d_0 = torch.squeeze(self.y_d_0, 0)
        modal_imgs.append(self.y_b_0)
        modal_imgs.append(self.y_d_0)
        modal_imgs.append(torch.squeeze(self.real_B,0))
        visual_ret = OrderedDict()
        for name, img in zip(self.modal_names, modal_imgs):
            visual_ret[name] = img
        visual_ret['fake_' + self.modal_names[-1]] = torch.squeeze(self.fake_B,0)

        return visual_ret

