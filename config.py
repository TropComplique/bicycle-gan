parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla | lsgan ｜ wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy: linear | step | plateau | cosine')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr_decay_iters', type=int, default=100, help='multiply by a gamma every lr_decay_iters iterations')
        # lambda parameters
        parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for |B-G(A, E(B))|')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight on D loss. D(G(A, E(B)))')
        parser.add_argument('--lambda_GAN2', type=float, default=1.0, help='weight on D2 loss, D(G(A, random_z))')
        parser.add_argument('--lambda_z', type=float, default=0.5, help='weight for ||E(G(random_z)) - random_z||')
        parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for KL loss')
        parser.add_argument('--use_same_D', action='store_true', help='if two Ds share the weights or not')

        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--nz', type=int, default=8, help='#latent vector')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
        parser.add_argument('--name', type=str, default='', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='not implemented')
        parser.add_argument('--dataset_mode', type=str, default='aligned', help='aligned,single')
        parser.add_argument('--model', type=str, default='bicycle_gan', help='chooses which model to use. bicycle,, ...')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads', default=4, type=int, help='# sthreads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')


        # extra parameters
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

NZ=8
NO_FLIP='--no_flip'
DIRECTION='AtoB'
LOAD_SIZE=256
CROP_SIZE=256
INPUT_NC=1
NITER=30
NITER_DECAY=30
