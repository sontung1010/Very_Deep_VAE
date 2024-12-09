HPARAMS_REGISTRY = {}

class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None
    def __setattr__(self, attr, value):
        self[attr] = value

flower32 = Hyperparams()
flower32.width = 384
flower32.lr = 0.0002
flower32.zdim = 16
flower32.wd = 0.01
flower32.dec_blocks = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
flower32.enc_blocks = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
flower32.warmup_iters = 100
flower32.dataset = 'flower32'
flower32.n_batch = 16
flower32.ema_rate = 0.9999
HPARAMS_REGISTRY['flower32'] = flower32

imagenet32 = Hyperparams()
imagenet32.width = 384
imagenet32.lr = 0.00015
imagenet32.zdim = 16
imagenet32.wd = 0.01
imagenet32.dec_blocks = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
imagenet32.enc_blocks = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
imagenet32.warmup_iters = 100
imagenet32.dataset = 'imagenet32'
imagenet32.n_batch = 16
imagenet32.ema_rate = 0.9999
HPARAMS_REGISTRY['imagenet32'] = imagenet32

cifar10 = Hyperparams()
cifar10.width = 384
cifar10.lr = 0.0002
cifar10.zdim = 16
cifar10.wd = 0.01
cifar10.dec_blocks = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
cifar10.enc_blocks = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
cifar10.warmup_iters = 100
cifar10.dataset = 'cifar10'
cifar10.n_batch = 16
cifar10.ema_rate = 0.9999
HPARAMS_REGISTRY['cifar10'] = cifar10


def parse_args_and_update_hparams(H, parser, s=None):
    args = parser.parse_args(s)
    valid_args = set(args.__dict__.keys())
    hparam_sets = [x for x in args.hparam_sets.split(',') if x]
    for hp_set in hparam_sets:
        hps = HPARAMS_REGISTRY[hp_set]
        for k in hps:
            if k not in valid_args:
                raise ValueError(f"{k} not in default args")
        parser.set_defaults(**hps)
    H.update(parser.parse_args(s).__dict__)


def add_vae_arguments(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--data_root', type=str, default='./')
    parser.add_argument('--desc', type=str, default='final_code')
    parser.add_argument('--hparam_sets', '--hps', type=str)
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--restore_ema_path', type=str, default=None)
    parser.add_argument('--restore_log_path', type=str, default=None)
    parser.add_argument('--restore_optimizer_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--ema_rate', type=float, default=0.999)
    parser.add_argument('--enc_blocks', type=str, default=None)
    parser.add_argument('--dec_blocks', type=str, default=None)
    parser.add_argument('--zdim', type=int, default=16)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--custom_width_str', type=str, default='')
    parser.add_argument('--bottleneck_multiple', type=float, default=0.25)
    parser.add_argument('--no_bias_above', type=int, default=64)
    parser.add_argument('--scale_encblock', action="store_true")
    parser.add_argument('--test_eval', type=bool, default = False)
    parser.add_argument('--warmup_iters', type=float, default=0)
    parser.add_argument('--num_mixtures', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=200.0)
    parser.add_argument('--skip_threshold', type=float, default=400.0)
    parser.add_argument('--lr', type=float, default=0.00015)
    parser.add_argument('--lr_prior', type=float, default=0.00015)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--wd_prior', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--iters_per_ckpt', type=int, default=2500)
    parser.add_argument('--iters_per_print', type=int, default=100)
    parser.add_argument('--iters_per_save', type=int, default=1000)
    parser.add_argument('--iters_per_images', type=int, default=1000)
    parser.add_argument('--epochs_per_eval', type=int, default=1)
    parser.add_argument('--epochs_per_probe', type=int, default=None)
    parser.add_argument('--epochs_per_eval_save', type=int, default=20)
    parser.add_argument('--num_images_visualize', type=int, default=8)
    parser.add_argument('--num_variables_visualize', type=int, default=6)
    parser.add_argument('--num_temperatures_visualize', type=int, default=3)
    return parser
