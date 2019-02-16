import torch


class BicycleGAN:

    def __init__(self, opt):
        assert opt.batch_size % 2 == 0  # load two images at one time.

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'D', 'G_GAN2', 'D2', 'G_L1', 'z_L1', 'kl']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A_encoded', 'real_B_encoded', 'fake_B_random', 'fake_B_encoded']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_D2 = opt.isTrain and opt.lambda_GAN2 > 0.0 and not opt.use_same_D
        use_E = opt.isTrain or not opt.no_encode
        use_vae = True
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample)
        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        if use_D2:
            self.model_names += ['D2']
            self.netD2 = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                           init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        if use_E:
            self.model_names += ['E']
            self.netE = networks.define_E(opt.output_nc, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, vaeLike=use_vae)


        self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionZ = torch.nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def is_train(self):
        """check if the current batch is good for training."""
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.to(self.device)

    def encode(self, input_image):
        mu, logvar = self.netE.forward(input_image)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    def test(self, z0=None, encode=False):
        with torch.no_grad():
            if encode:  # use encoded z
                z0, _ = self.netE(self.real_B)
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
            self.fake_B = self.netG(self.real_A, z0)
            return self.real_A, self.fake_B, self.real_B

    def forward(self):
        # get real images
        half_size = self.opt.batch_size // 2
        # A1, B1 for encoded; A2, B2 for random
        self.real_A_encoded = self.real_A[0:half_size]
        self.real_B_encoded = self.real_B[0:half_size]
        self.real_B_random = self.real_B[half_size:]
        # get encoded z
        self.z_encoded, self.mu, self.logvar = self.encode(self.real_B_encoded)
        # get random z
        self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.opt.nz)
        # generate fake_B_encoded
        self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded)
        # generate fake_B_random
        self.fake_B_random = self.netG(self.real_A_encoded, self.z_random)

        self.fake_data_encoded = self.fake_B_encoded
        self.fake_data_random = self.fake_B_random
        self.real_data_encoded = self.real_B_encoded
        self.real_data_random = self.real_B_random

        # compute z_predict
        if self.opt.lambda_z > 0.0:
            self.mu2, logvar2 = self.netE(self.fake_B_random)  # mu2 is a point estimate

    def backward_D(self, netD, real, fake):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake.detach())
        # real
        pred_real = netD(real)
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake = netD(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD, self.opt.lambda_GAN)
        if self.opt.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD, self.opt.lambda_GAN2)
        else:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2, self.opt.lambda_GAN2)
        # 2. KL loss
        if self.opt.lambda_kl > 0.0:
            self.loss_kl = torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * (-0.5 * self.opt.lambda_kl)
        else:
            self.loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_kl
        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        self.set_requires_grad(self.netD, True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded)
            if self.opt.use_same_D:
                self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random, self.fake_data_random)
            self.optimizer_D.step()

        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(self.netD2, self.real_data_random, self.fake_data_random)
            self.optimizer_D2.step()

    def backward_G_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = torch.mean(torch.abs(self.mu2 - self.z_random)) * self.opt.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad(self.netD, False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()
        self.optimizer_E.step()
        # update G only
        if self.opt.lambda_z > 0.0:
            self.optimizer_G.zero_grad()
            self.optimizer_E.zero_grad()
            self.backward_G_alone()
            self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G_and_E()
        self.update_D()


import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call  `BaseModel.__init__(self, opt)`
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def is_train(self):
        """check if the current batch is good for training."""
        return True

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
