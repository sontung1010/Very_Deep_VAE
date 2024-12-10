import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# Function to create a mapping of resolution to width settings
def from_parameter_get_width(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping

def get_conv(input_dimension, output_dimension, kernel_size, stride, padding, groups=1, scaled=False, zero_bias=True, zero_weights=False):
    convolution = nn.Conv2d(input_dimension, output_dimension, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        convolution.bias.data *= 0.0
    if zero_weights:
        convolution.weight.data *= 0.0
    return convolution

## This is defining convolutions layer for base blocks
# =========================================================
def convolution_3x3(input_dim, output_dim, groups=1, scaled=False, zero_bias=True, zero_weights=False):
    convolution_3x3 = get_conv(input_dim, output_dim, 3, 1, 1, groups=groups, scaled=scaled, zero_bias = zero_bias, zero_weights = zero_weights)
    return convolution_3x3


def convolution_1x1(input_dim, output_dim, groups=1, scaled=False, zero_bias=True, zero_weights=False):
    convolution_1x1 = get_conv(input_dim, output_dim, 1, 1, 0, groups=groups, scaled=scaled, zero_bias = zero_bias, zero_weights = zero_weights)
    return convolution_1x1


## Code below here is mainly for calculating the probalistic setting in the model.
# High complexity in the code.
# We studied the python code from original research (VDVAE) and tried to reproduce based on the original code.
# ==================================================================================

@torch.jit.script
def sample_diag_gaussian(mu, logsigma):
    epsilon = torch.empty_like(mu)
    epsilon.normal_(0., 1.)
    stddev = torch.exp(logsigma)
    return_value = mu + stddev * epsilon
    return return_value

@torch.jit.script
def compute_gaussian_kl(mu1, mu2, log_sigma1, log_sigma2):
    return -0.5 + log_sigma2 - log_sigma1 + 0.5 * (log_sigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (log_sigma2.exp() ** 2)

def log_prob_from_logits(x):
    axis = len(x.shape) - 1
    max_val = x.max(dim=axis, keepdim=True)[0]
    centered_x = x - max_val
    return centered_x - torch.log(torch.exp(centered_x).sum(dim=axis, keepdim=True))


def constant_max(t, constant):
    other = torch.ones_like(t) * constant
    return_max = torch.max(t, other)
    return return_max

def constant_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


## implementation of the loss calculation
# =======================================================
def discretized_mix_logistic_loss(x, l, low_bit=False):
    ## extract the shape of the input tensors
    xs = []
    for s in x.shape:
        xs.append(s)
    #print(xs)
    ls = []
    for s in l.shape:
        ls.append(s) 
    #print(ls) 
    nr_mix = int(ls[-1] / 10) # each have 10
    # Separate logits for mixture weights
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])

    # Extract parameters for the logistic mixture
    means = l[:, :, :, :, :nr_mix]
    log_scales = l[:, :, :, :, nr_mix:2 * nr_mix]
    log_scales = constant_max(log_scales, -7.)
    #print(log_scales)
    coeffs = l[:, :, :, :, 2 * nr_mix:3 * nr_mix]
    coeffs = torch.tanh(coeffs) 

    x_reshaped = torch.reshape(x, xs + [1])
    zeros_like_x = torch.zeros(xs + [nr_mix]).to(x.device)
    x = x_reshaped + zeros_like_x

    # Compute means for second and third channels
    m2 = means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]
    m2 = torch.reshape(m2, [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]
    m3 = torch.reshape(m3, [xs[0], xs[1], xs[2], 1, nr_mix])

    # Combine means for all channels
    means_channel0 = torch.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    means = torch.cat([means_channel0, m2, m3], dim=3)
    centered_x = x - means # centering from x_means
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
    log_cdf_plus = plus_in - F.softplus(plus_in) # log(sigmoid(plus_in))
    log_one_minus_cdf_min = -F.softplus(min_in)# log(1 - sigmoid(min_in))
    #print(type(log_one_minus_cdf_min))
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # Compute log probabilities
    if low_bit:
        log_probs = torch.where(x < -0.999,log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min,
                torch.where(cdf_delta > 1e-5, torch.log(constant_max(cdf_delta, 1e-12)), log_pdf_mid - np.log(15.5))))
    else:
        log_probs = torch.where(
            x < -0.999, log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min,
                torch.where(
                    cdf_delta > 1e-5, torch.log(constant_max(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))
    # Sum log probabilities alonb channels
    log_probs = log_probs.sum(dim=3)
    log_probs = log_probs + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, -1)
    # Final loss calculation, negative log likelihood
    final_loss = -1. * mixture_probs.sum(dim=[1, 2]) / np.prod(xs[1:])
    return final_loss


## Function below is for sampling the model from mixture of logistic distribution
# widely used in image generation.
# =======================================================
def sample_from_discretized_mix_logistic(l, nr_mix):
    # Extract the shape of the input tensor
    ls = []
    for s in l.shape:
        ls.append(s)
    # Dshape for RGB output images
    xs = ls[:-1] + [3]
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])

    # mixture indices <- Gumbel-Softmax trick
    eps = torch.empty(logit_probs.shape, device=l.device).uniform_(1e-5, 1. - 1e-5)
    gumbel_noise = -torch.log(-torch.log(eps))
    gumbel_logits = logit_probs - gumbel_noise
    amax = torch.argmax(gumbel_logits, dim=3)
    sel = F.one_hot(amax, num_classes=nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])

    means_raw = l[:, :, :, :, :nr_mix]
    means = (means_raw * sel).sum(dim=4)
    log_scales_raw = l[:, :, :, :, nr_mix:nr_mix * 2]
    log_scales = (log_scales_raw * sel).sum(dim=4)
    log_scales = constant_max(log_scales, -7.)

    coeffs_raw = l[:, :, :, :, nr_mix * 2:nr_mix * 3]
    coeffs = (torch.tanh(coeffs_raw) * sel).sum(dim=4)

    # print(coeffs)
    # print(type(coeffs))
    u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1. - 1e-5)
    logistic_noise = torch.log(u) - torch.log(1. - u)
    x = means + torch.exp(log_scales) * logistic_noise
    x0_clamped = constant_min(constant_max(x[:, :, :, 0], -1.), 1.)
    x1_clamped = constant_min(constant_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0_clamped, -1.), 1.)
    x2_clamped = constant_min(constant_max(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0_clamped + coeffs[:, :, :, 2] * x1_clamped, -1.), 1.)

    x0_final = torch.reshape(x0_clamped, xs[:-1] + [1])
    x1_final = torch.reshape(x1_clamped, xs[:-1] + [1])
    x2_final = torch.reshape(x2_clamped, xs[:-1] + [1])
    result = torch.cat([x0_final, x1_final, x2_final], dim=3)
    return result

# This is tool to process string in the hyperparameters
def prepare_string(s):
    layers = [] 
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers




