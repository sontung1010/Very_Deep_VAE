import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu


def get_conv(input_dim, output_dim, kernel_size, stride, padding, groups=1, scaled=False, zero_bias=True, zero_weights=False):
    convolution = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:convolution.bias.data *= 0.0
    if zero_weights:convolution.weight.data *= 0.0
    return convolution


def get_3x3(input_dim, output_dim, groups=1, scaled=False, zero_bias=True, zero_weights=False):
    convolution_3x3 = get_conv(input_dim, output_dim, 3, 1, 1, groups=groups, scaled=scaled, zero_bias = zero_bias, zero_weights = zero_weights)
    return convolution_3x3


def get_1x1(input_dim, output_dim, groups=1, scaled=False, zero_bias=True, zero_weights=False):
    convolution_1x1 = get_conv(input_dim, output_dim, 1, 1, 0, groups=groups, scaled=scaled, zero_bias = zero_bias, zero_weights = zero_weights)
    return convolution_1x1


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def const_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


def discretized_mix_logistic_loss(x, l, low_bit=False):
    # This portion of the code was written with reference to the following GitHub.
    # https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    xs = [s for s in x.shape]  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = [s for s in l.shape]  # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10)  # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = const_max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = torch.reshape(x, xs + [1]) + torch.zeros(xs + [nr_mix]).to(x.device)
    m2 = torch.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = torch.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    means = torch.cat([torch.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    if low_bit:
        plus_in = inv_stdv * (centered_x + 1. / 31.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 31.)
    else:
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)  
    log_one_minus_cdf_min = -F.softplus(min_in)  
    cdf_delta = cdf_plus - cdf_min 
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)  

    if low_bit:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(15.5))))
    else:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(127.5))))
    log_probs = log_probs.sum(dim=3) + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, -1)
    return -1. * mixture_probs.sum(dim=[1, 2]) / np.prod(xs[1:])


def sample_from_discretized_mix_logistic(l, nr_mix):
    ls = [s for s in l.shape]
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    eps = torch.empty(logit_probs.shape, device=l.device).uniform_(1e-5, 1. - 1e-5)
    amax = torch.argmax(logit_probs - torch.log(-torch.log(eps)), dim=3)
    sel = F.one_hot(amax, num_classes=nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    log_scales = const_max((l[:, :, :, :, nr_mix:nr_mix * 2] * sel).sum(dim=4), -7.)
    coeffs = (torch.tanh(l[:, :, :, :, nr_mix * 2:nr_mix * 3]) * sel).sum(dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = const_min(const_max(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
    return torch.cat([torch.reshape(x0, xs[:-1] + [1]), torch.reshape(x1, xs[:-1] + [1]), torch.reshape(x2, xs[:-1] + [1])], dim=3)

# Function to create a mapping of resolution to width settings
def get_width_settings(width, s):
    # Default mapping assigns the same width to all resolutions
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',') # If specific settings are provided, parse and update the mapping
        for ss in s:
            k, v = ss.split(':')  # Syntax: resolution:width
            mapping[int(k)] = int(v)
    return mapping

# Function to parse a string and define layer settings
def parse_layer_string(s):
    layers = []  # List to store parsed layer settings
    for ss in s.split(','):
        if 'x' in ss:
            # Syntax: resxnum (e.g., 32x2 means two layers with resolution 32)
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            # Syntax: resmMixin (e.g., 32m2 adds a mixin)
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            # Syntax: resdDownRate (e.g., 32d2 applies down-sampling with rate 2)
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            # Single resolution case (e.g., "32" adds a single layer with resolution 32)
            res = int(ss)
            layers.append((res, None))
    return layers




