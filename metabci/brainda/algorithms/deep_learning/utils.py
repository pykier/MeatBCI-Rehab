import torch.nn as nn
import torch
import scipy.signal as signal
import numpy as np
import copy
import math
import random
from torch.autograd import Function
from torch.nn import functional as F
from torch.functional import Tensor
from typing import Callable, Tuple
from typing import Any

dtype=torch.float32

EPS = {torch.float32: 1e-6, torch.float64: 1e-9}

def qr_algorithm(A, max_iter=5, tol=1e-3):
    n = A.shape[1]
    A_k = A.clone().to(A.device)

    Q_total = torch.eye(n, device=A.device)
    for i in range(max_iter):
        # QR 分解
        Q, R = torch.linalg.qr(A_k)
        A_k = R @ Q
        Q_total = Q_total @ Q
        # 判断收敛性
        # off_diag_sum = torch.sum(abs(A_k - torch.diag_embed(torch.diagonal(A_k,dim1=1,dim2=2))))
        # if off_diag_sum < tol:
        #     break
    # s=torch.diagonal(A_k,dim1=1,dim2=2)
    # U=Q_total
    # res=U @ torch.diag_embed(s) @ U.mT
    return torch.diagonal(A_k,dim1=1,dim2=2), Q_total

def ensure_sym(A: Tensor) -> Tensor:
    """Ensures that the last two dimensions of the tensor are symmetric.
    Parameters
    ----------
    A : torch.Tensor
        with the last two dimensions being identical
    -------
    Returns : torch.Tensor
    """
    return 0.5 * (A + A.transpose(-1,-2))


class StiefelParameter(nn.Parameter):
    """A kind of Variable that is to be considered a module parameter on the space of
        Stiefel manifold.
    """
    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)





class filterBank(object):
    """
    filter the given signal in the specific bands using cheby2 iir filtering.
    If only one filter is specified then it acts as a simple filter and returns 2d matrix
    Else, the output will be 3d with the filtered signals appended in the third dimension.
    axis is the time dimension along which the filtering will be applied
    """

    def __init__(self, filtBank=[[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]], fs=250, filtAllowance=2, axis=-1, filtType='filter'):
        self.filtBank = filtBank
        self.fs = fs
        self.filtAllowance = filtAllowance
        self.axis = axis
        self.filtType = filtType

    def bandpassFilter(self, data, bandFiltCutF, fs, filtAllowance=2, axis=-1, filtType='filter'):
        """
         Filter a signal using cheby2 iir filtering.

        Parameters
        ----------
        data: 2d/ 3d np array
            trial x channels x time
        bandFiltCutF: two element list containing the low and high cut off frequency in hertz.
            if any value is specified as None then only one sided filtering will be performed
        fs: sampling frequency
        filtAllowance: transition bandwidth in hertz
        filtType: string, available options are 'filtfilt' and 'filter'

        Returns
        -------
        dataOut: 2d/ 3d np array after filtering
            Data after applying bandpass filter.
        """
        aStop = 30  # stopband attenuation
        aPass = 3  # passband attenuation
        nFreq = fs / 2  # Nyquist frequency

        if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (
                bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data

        elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            fPass = bandFiltCutF[1] / nFreq
            fStop = (bandFiltCutF[1] + filtAllowance) / nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'lowpass')

        elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            fPass = bandFiltCutF[0] / nFreq
            fStop = (bandFiltCutF[0] - filtAllowance) / nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'highpass')

        else:
            # band-pass filter
            # print("Using bandpass filter")
            fPass = (np.array(bandFiltCutF) / nFreq).tolist()
            fStop = [(bandFiltCutF[0] - filtAllowance) / nFreq, (bandFiltCutF[1] + filtAllowance) / nFreq]
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'bandpass')

        if filtType == 'filtfilt':
            dataOut = signal.filtfilt(b, a, data, axis=axis)
        else:
            dataOut = signal.lfilter(b, a, data, axis=axis)
        return dataOut

    def __call__(self, data1):

        data = copy.deepcopy(data1)
        d = data

        # initialize output
        out = np.zeros([*d.shape, len(self.filtBank)])

        # repetitively filter the data.
        for i, filtBand in enumerate(self.filtBank):
            out[:, ..., i] = self.bandpassFilter(d, filtBand, self.fs, self.filtAllowance,
                                               self.axis, self.filtType)


        data = torch.from_numpy(out).float()
        return data

class SUMlayer(nn.Module):
    def forward(self, *x):

        return sum(*x)
    def __repr__(self): return f'{self.__class__.__name__}'


class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim
    def forward(self, *x): return torch.cat(*x, dim=self.dim)
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'

class Patch(nn.Module):
    def __init__(self,seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
        tgt_len = patch_len  + stride*(self.num_patch-1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x n_vars]
        """
        x = x[:, self.s_begin:, :]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)                 # xb: [bs x num_patch x n_vars x patch_len]
        return x


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class LayerNormalization(nn.Module):

    def __init__(self,
                 normal_shape,
                 gamma=True,
                 beta=True,
                 epsilon=1e-6):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super(LayerNormalization, self).__init__()
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        if gamma:
            self.gamma = nn.Parameter(torch.ones(*normal_shape))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.Parameter(torch.zeros(*normal_shape))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (torch.sqrt(var + 1e-6))
        if self.gamma is not None:
            x *= self.gamma.expand_as(x)
        if self.beta is not None:
            x += self.beta.expand_as(x)
        return x

    def extra_repr(self):
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
            self.normal_shape, self.gamma is not None, self.beta is not None, self.epsilon,
        )


class Stie_W(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(Stie_W, self).__init__()
        assert  input_dim>=output_dim
        self.output_dim=output_dim
        geoopt_used=False
        # if geoopt_used:
        #     manifold = geoopt.Stiefel(canonical=False)
        #     self.weight = geoopt.ManifoldParameter(manifold.random(input_dim, output_dim), manifold=manifold,
        #                                            requires_grad=True)
        # else:
        self.weight = StiefelParameter(torch.FloatTensor(input_dim,output_dim), requires_grad=True)
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        B,C,H,W=x.shape
        input=x.reshape(B,C,H*W)
        output=torch.matmul(self.weight.t(),input)
        output=output.reshape(B,self.output_dim,H,W)
        return output


class PointwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointwiseConv2d,self).__init__()
        # 初始化权重和偏置
        self.weight = StiefelParameter(torch.FloatTensor(in_channels,out_channels), requires_grad=True)
        self.bias =  None
        self.stride = 1
        self.padding = 0
        self.in_chan=in_channels
        self.out_chan=out_channels
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        # 使用F.conv2d进行卷积操作
        weight=self.weight.t().reshape(self.out_chan,self.in_chan,1,1)
        out= F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding)
        return out

class PointwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 初始化权重和偏置
        self.weight = StiefelParameter(torch.FloatTensor(in_channels,out_channels), requires_grad=True)
        self.bias =  None
        self.stride = 1
        self.padding = 0
        self.in_chan=in_channels
        self.out_chan=out_channels
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        # 使用F.conv2d进行卷积操作
        weight=self.weight.t().reshape(self.out_chan,self.in_chan,1)
        out= F.conv1d(x, weight, self.bias, stride=self.stride, padding=self.padding)
        return out

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = StiefelParameter(torch.FloatTensor(in_features,out_features), requires_grad=True)
        self.bias = None
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        # 使用 F.linear 进行线性变换
        weight=self.weight.t()
        return F.linear(x, weight, self.bias)


class sym_modeig:
    """Basic class that modifies the eigenvalues with an arbitrary elementwise function
    """

    @staticmethod
    def forward(M : Tensor, fun : Callable[[Tensor], Tensor], fun_param : Tensor = None,
                ensure_symmetric : bool = False, ensure_psd : bool = False) -> Tensor:
        """Modifies the eigenvalues of a batch of symmetric matrices in the tensor M (last two dimensions).

        Source: Brooks et al. 2019, Riemannian batch normalization for SPD neural networks, NeurIPS

        Parameters
        ----------
        M : torch.Tensor
            (batch) of symmetric matrices
        fun : Callable[[Tensor], Tensor]
            elementwise function
        ensure_symmetric : bool = False (optional)
            if ensure_symmetric=True, then M is symmetrized
        ensure_psd : bool = False (optional)
            if ensure_psd=True, then the eigenvalues are clamped so that they are > 0
        -------
        Returns : torch.Tensor with modified eigenvalues
        """
        # if ensure_symmetric:
        #     M = ensure_sym(M)

        # compute the eigenvalues and vectors
        # U, s, vt = torch.linalg.svd(M)
        s,U = qr_algorithm(M)
        if ensure_psd:
            s = s.clamp(min=EPS[s.dtype])
        # modify the eigenvalues
        smod = fun(s, fun_param)
        X = U @ torch.diag_embed(smod) @ U.transpose(-1,-2)

        return X, s, smod, U

    @staticmethod
    def backward(dX : Tensor, s : Tensor, smod : Tensor, U : Tensor,
                    fun_der : Callable[[Tensor], Tensor], fun_der_param : Tensor = None) -> Tensor:
        """Backpropagates the derivatives

        Source: Brooks et al. 2019, Riemannian batch normalization for SPD neural networks, NeurIPS

        Parameters
        ----------
        dX : torch.Tensor
            (batch) derivatives that should be backpropagated
        s : torch.Tensor
            eigenvalues of the original input
        smod : torch.Tensor
            modified eigenvalues
        U : torch.Tensor
            eigenvector of the input
        fun_der : Callable[[Tensor], Tensor]
            elementwise function derivative
        -------
        Returns : torch.Tensor containing the backpropagated derivatives
        """

        # compute Lowener matrix
        # denominator
        L_den = s[...,None] - s[...,None].transpose(-1,-2)
        # find cases (similar or different eigenvalues, via threshold)
        is_eq = L_den.abs() < EPS[s.dtype]
        L_den[is_eq] = 1.0
        # case: sigma_i != sigma_j
        L_num_ne = smod[...,None] - smod[...,None].transpose(-1,-2)
        L_num_ne[is_eq] = 0
        # case: sigma_i == sigma_j
        sder = fun_der(s, fun_der_param)
        L_num_eq = 0.5 * (sder[...,None] + sder[...,None].transpose(-1,-2))
        L_num_eq[~is_eq] = 0
        # compose Loewner matrix
        L = (L_num_ne + L_num_eq) / L_den
        dM = U @  (L * (U.transpose(-1,-2) @ ensure_sym(dX) @ U)) @ U.transpose(-1,-2)
        return dM


class sym_logm(Function):
    """
    Computes the matrix logarithm for a batch of SPD matrices.
    Ensures that the input matrices are SPD by clamping eigenvalues.
    During backprop, the update along the clamped eigenvalues is zeroed
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        # ensure that the eigenvalues are positive
        return s.clamp(min=EPS[s.dtype]).log()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        # compute derivative
        sder = s.reciprocal()
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_logm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_logm.derivative), None


class LogmLayer(nn.Module):

    def __init__(self, input_size, vectorize=False):
        super(LogmLayer, self).__init__()
        self.vectorize = vectorize

    def forward(self, input):
        output=sym_logm.apply(input)
        return output
import torch
import torch.nn as nn
import torch.nn.functional as F

# SE Attention 模块
class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.channel_excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# 原始 R_attention 替换为 SE_Attention 后的新模块
# class R_attention_SE(nn.Module):
#     '''
#     If these codes help you, please cite our paper:
#
#     Liang, W., Allison, B. Z., Xu, R., He, X., Wang, X., Cichocki, A., & Jin, J. (2025).
#     SecNet: A second order neural network for MI-EEG. Information Processing & Management, 62(3), 104012.
#     '''
#     def __init__(self, dim, drop_att=0.1):
#         super(R_attention_SE, self).__init__()
#         self.Flag_atten = True  # 是否使用注意力
#         self.Flag_pe = True     # 是否使用位置编码
#         self.drop = nn.Dropout(drop_att)
#         self.attention = SEAttention(channel=dim)  # 使用 SE Attention
#         self.PE = PositionalEmbedding(dim)         # 保留原有位置编码
#
#     def forward(self, h_in):
#         b, c, _, M = h_in.size()
#
#         # 构造 I_hat 矩阵
#         I_hat = (-1. / M / M) * torch.ones(M, M, device=h_in.device) + (1. / M) * torch.eye(M, M, device=h_in.device)
#         I_hat = I_hat.view(1, M, M).repeat(b, 1, 1).type(h_in.dtype)
#
#         # 添加位置编码（如果启用）
#         if self.Flag_pe:
#             pe = self.PE(h_in)
#             h_in = h_in + pe
#
#         # 应用 SE 注意力机制
#         if self.Flag_atten:
#             hb = h_in
#             h_in = self.attention(h_in)  # 使用 SE 注意力
#             h_in = self.drop(h_in)
#
#         # 计算最终输出矩阵
#         h = h_in.squeeze(2) @ I_hat @ h_in.squeeze(2).mT \
#             + hb.squeeze(2) @ I_hat @ hb.squeeze(2).mT \
#             + torch.eye(hb.size(1), device=h_in.device) * 1e-8
#
#         return h


class R_attention(nn.Module):
    '''
    If these codes help you, please cite our paper:

    Liang, W., Allison, B. Z., Xu, R., He, X., Wang, X., Cichocki, A., & Jin, J. (2025).  SecNet: A second order neural network for MI-EEG.  *Information Processing & Management*, 62(3), 104012.
    '''
    def __init__(self, dim, k_size, drop_att=0.1):
        super(R_attention, self).__init__()
        self.Flag_atten = True # whether to use attention
        self.Flag_pe = True # whether to use positional encoding
        self.drop=nn.Dropout(drop_att)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(dim, dim, kernel_size=k_size, groups=dim)
        self.sigmoid = nn.Sigmoid()
        self.PE=PositionalEmbedding(dim)


    def forward(self,h_in):
        b, c, _, M = h_in.size()
        I_hat = (-1. / M / M) * torch.ones(M, M, device=h_in.device) + (1. / M) * torch.eye(M, M, device=h_in.device)
        I_hat = I_hat.view(1, M, M).repeat(b, 1, 1).type(h_in.dtype)
        # print('I_hat',I_hat.shape)
        if self.Flag_atten:
            hb=h_in
            if self.Flag_pe:
                pe = self.PE(h_in)
                h_in=h_in+pe

            y = self.avg_pool(h_in)
            y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size),
                                     padding=(0, (self.k_size - 1) // 2))
            y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
            y = self.sigmoid(y)
            h = self.drop(h_in * y.expand_as(h_in))
            h=h.squeeze(2) @ I_hat @ h.squeeze(2).mT + hb.squeeze(2) @I_hat@ hb.squeeze(2).mT + torch.eye(hb.size(1),device=hb.device)*1e-8
        else:
            if self.Flag_pe:
                pe = self.PE(h_in)
                h_in+=pe

            h = h_in.squeeze(2) @I_hat@ h_in.squeeze(2).mT + torch.eye(h_in.size(1), device=h_in.device) * 1e-8
        return h

#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
#
#
# class R_attention(nn.Module):
#     '''
#     If these codes help you, please cite our paper:
#     Liang, W., et al. (2025). SecNet: A second order neural network for MI-EEG.
#     Information Processing & Management, 62(3), 104012.
#     '''
#
#     def __init__(self, dim, k_size, drop_att=0.1, num_heads=4):
#         super(R_attention, self).__init__()
#         assert dim % num_heads == 0, "dim must be divisible by num_heads"
#
#         self.Flag_atten = True
#         self.Flag_pe = True
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.dim = dim
#         self.k_size = k_size
#
#         self.drop = nn.Dropout(drop_att)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(dim, dim, kernel_size=k_size, groups=dim)
#         self.sigmoid = nn.Sigmoid()
#         self.PE = PositionalEmbedding(dim)
#
#     def forward(self, h_in):
#         # h_in: [B, C, 1, M]
#         b, c, _, M = h_in.size()
#
#         I_hat = (-1. / M / M) * torch.ones(M, M, device=h_in.device) + (1. / M) * torch.eye(M, M, device=h_in.device)
#         I_hat = I_hat.view(1, M, M).repeat(b, 1, 1).type(h_in.dtype)  # [B, M, M]
#
#         if self.Flag_atten:
#             hb = h_in  # residual
#             if self.Flag_pe:
#                 pe = self.PE(h_in)
#                 h_in = h_in + pe
#
#             y = self.avg_pool(h_in)  # [B, C, 1, 1]
#             y = F.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size),
#                          padding=(0, (self.k_size - 1) // 2))  # [B, C*k_size, 1]
#             y = self.conv(y.transpose(-1, -2))  # [B, C, 1]
#             y = self.sigmoid(y).unsqueeze(-1)  # [B, C, 1, 1]
#             h_weighted = self.drop(h_in * y.expand_as(h_in))  # [B, C, 1, M]
#
#             h_cov = torch.zeros(b, c, c, device=h_in.device, dtype=h_in.dtype)  # final output [B, C, C]
#
#             # Split into heads and compute head-wise covariances
#             for i in range(self.num_heads):
#                 start = i * self.head_dim
#                 end = (i + 1) * self.head_dim
#                 h_i = h_weighted[:, start:end, 0, :]  # [B, head_dim, M]
#                 hb_i = hb[:, start:end, 0, :]  # [B, head_dim, M]
#
#                 cov_i = h_i @ I_hat @ h_i.transpose(1, 2) + hb_i @ I_hat @ hb_i.transpose(1,
#                                                                                           2)  # [B, head_dim, head_dim]
#                 cov_i = cov_i + torch.eye(self.head_dim, device=h_in.device) * 1e-8  # stability
#
#                 h_cov[:, start:end, start:end] = cov_i  # insert into corresponding block
#
#             return h_cov  # [B, C, C]
#
#         else:
#             if self.Flag_pe:
#                 h_in += self.PE(h_in)
#
#             h_raw = h_in.squeeze(2)  # [B, C, M]
#             h_cov = h_raw @ I_hat @ h_raw.transpose(1, 2) + torch.eye(c, device=h_in.device) * 1e-8  # [B, C, C]
#             return h_cov


# class R_attention(nn.Module):
#     '''
#     If these codes help you, please cite our paper:
#
#     Liang, W., Allison, B. Z., Xu, R., He, X., Wang, X., Cichocki, A., & Jin, J. (2025).  SecNet: A second order neural network for MI-EEG.  *Information Processing & Management*, 62(3), 104012.
#     '''
#
#     def __init__(self, dim, k_size, drop_att=0.1):
#         super(R_attention, self).__init__()
#         self.Flag_atten = True
#         self.Flag_pe = True
#         self.drop = nn.Dropout(drop_att)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.k_size = k_size
#         self.conv = nn.Conv1d(dim, dim, kernel_size=k_size, groups=dim)
#         self.sigmoid = nn.Sigmoid()
#         self.PE = PositionalEmbedding(dim)
#
#         # 可学习的I_hat系数
#         self.alpha = nn.Parameter(torch.tensor(1.0))  # 原全1矩阵的系数
#         self.beta = nn.Parameter(torch.tensor(1.0))  # 原单位矩阵的系数
#
#         # 动态权重参数
#         self.gamma = nn.Parameter(torch.ones(2))  # 用于平衡h和hb的协方差
#
#         # 通道注意力模块
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Conv1d(dim, dim // 4, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv1d(dim // 4, dim, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, h_in):
#         b, c, _, M = h_in.size()
#
#         # 生成动态I_hat矩阵
#         I_hat = (self.alpha * (-1.0 / (M * M)) * torch.ones(M, M, device=h_in.device)
#                  + self.beta * (1.0 / M) * torch.eye(M, device=h_in.device))
#         I_hat = I_hat.view(1, M, M).repeat(b, 1, 1).type(h_in.dtype)
#
#         if self.Flag_atten:
#             hb = h_in
#             if self.Flag_pe:
#                 pe = self.PE(h_in)
#                 h_in = h_in + pe
#
#             # 注意力机制处理
#             y = self.avg_pool(h_in)
#             y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size),
#                                      padding=(0, (self.k_size - 1) // 2))
#             y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
#             y = self.sigmoid(y)
#             h = self.drop(h_in * y.expand_as(h_in))
#
#             # 均值中心化处理
#             h_sqz = h.squeeze(2)  # [b,c,M]
#             h_mean = h_sqz.mean(dim=2, keepdim=True)
#             h_centered = h_sqz - h_mean
#
#             # 通道注意力加权
#             se_weights = self.se(h_centered)  # [b,c,1]
#             h_weighted = h_centered * se_weights  # [b,c,M]
#
#             # 计算h部分的协方差
#             h_part = torch.bmm(torch.bmm(h_weighted, I_hat), h_weighted.transpose(1, 2))
#
#             # 处理原始特征hb
#             hb_sqz = hb.squeeze(2)
#             hb_centered = hb_sqz - hb_sqz.mean(dim=2, keepdim=True)
#             hb_part = torch.bmm(torch.bmm(hb_centered, I_hat), hb_centered.transpose(1, 2))
#
#             # 动态权重融合
#             gamma = torch.softmax(self.gamma, dim=0)
#             h_cov = gamma[0] * h_part + gamma[1] * hb_part
#
#             # 正则化项
#             h_cov += torch.eye(c, device=h_in.device) * 1e-8
#
#         else:
#             if self.Flag_pe:
#                 pe = self.PE(h_in)
#                 h_in += pe
#
#             h_sqz = h_in.squeeze(2)
#             h_centered = h_sqz - h_sqz.mean(dim=2, keepdim=True)
#             h_cov = torch.bmm(torch.bmm(h_centered, I_hat), h_centered.transpose(1, 2))
#             h_cov += torch.eye(c, device=h_in.device) * 1e-8
#
#         return h_cov
# import torch
# import torch.nn as nn
#
#
# class R_attention(nn.Module):
#     '''
#     If these codes help you, please cite our paper:
#
#     Liang, W., Allison, B. Z., Xu, R., He, X., Wang, X., Cichocki, A., & Jin, J. (2025).  SecNet: A second order neural network for MI-EEG.  *Information Processing & Management*, 62(3), 104012.
#     '''
#
#     def __init__(self, dim, k_size, drop_att=0.1, window_size=31, stride=31):
#         super(R_attention, self).__init__()
#         self.Flag_atten = True  # 是否使用注意力机制
#         self.Flag_pe = True  # 是否使用位置编码
#         self.drop = nn.Dropout(drop_att)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.k_size = k_size
#         self.conv = nn.Conv1d(dim, dim, kernel_size=k_size, groups=dim)
#         self.sigmoid = nn.Sigmoid()
#         self.PE = PositionalEmbedding(dim)
#
#         # 新增滑动窗口参数
#         self.window_size = window_size
#         self.stride = stride
#
#     def forward(self, h_in):
#         b, c, _, M = h_in.size()
#
#         # 构造 I_hat 矩阵
#         I_hat = (-1. / M / M) * torch.ones(M, M, device=h_in.device) + (1. / M) * torch.eye(M, M, device=h_in.device)
#         I_hat = I_hat.view(1, M, M).repeat(b, 1, 1).type(h_in.dtype)
#
#         if self.Flag_atten:
#             hb = h_in
#             if self.Flag_pe:
#                 pe = self.PE(h_in)
#                 h_in = h_in + pe
#
#             # 注意力机制部分
#             y = self.avg_pool(h_in)
#             y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size),
#                                      padding=(0, (self.k_size - 1) // 2))
#             y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
#             y = self.sigmoid(y)
#             h = self.drop(h_in * y.expand_as(h_in))
#
#             # 原始协方差矩阵计算
#             cov_matrix_static = h.squeeze(2) @ I_hat @ h.squeeze(2).mT + \
#                                 hb.squeeze(2) @ I_hat @ hb.squeeze(2).mT + \
#                                 torch.eye(hb.size(1), device=hb.device) * 1e-8
#
#             # 动态协方差矩阵计算（滑动窗口）
#             dynamic_cov = self.compute_dynamic_covariance(h_in, window_size=31, stride=31, I_hat=I_hat)
#
#             # 将静态协方差矩阵与动态协方差矩阵相加
#             h = cov_matrix_static+dynamic_cov
#
#         else:
#             if self.Flag_pe:
#                 pe = self.PE(h_in)
#                 h_in += pe
#
#             # 原始协方差矩阵计算
#             h = h_in.squeeze(2) @ I_hat @ h_in.squeeze(2).mT + \
#                 torch.eye(h_in.size(1), device=h_in.device) * 1e-8
#
#         return h
#
#     def compute_dynamic_covariance(self, x, window_size, stride, I_hat):
#         """
#         计算滑动窗口动态协方差矩阵，每个窗口的计算方式与 hb.squeeze(2) @ I_hat @ hb.squeeze(2).mT 一致。
#
#         参数:
#         x: 输入张量，形状为 (batch_size, channels, 1, time_steps)
#         window_size: 滑动窗口大小
#         stride: 滑动窗口的步长
#         I_hat: 预先构造的矩阵，形状为 (batch_size, M, M)
#
#         返回:
#         dynamic_cov: 动态协方差矩阵，形状为 (batch_size, channels, channels)
#         """
#         batch_size, channels, _, time_steps = x.shape
#         windows = []
#
#         # 划分滑动窗口
#         for t in range(0, time_steps - window_size + 1, stride):
#             window = x[:, :, :, t:t + window_size]  # (batch_size, channels, 1, window_size)
#             windows.append(window)  # (batch_size, channels, window_size)
#
#         num_windows = len(windows)
#         cov_matrices = []
#
#         # 计算每个窗口的协方差矩阵
#         for window in windows:
#             # 确保窗口数据形状为 (batch_size, channels, M)，其中 M 是窗口大小
#             b, c, _,M = window.shape
#             pe = self.PE(window)
#             window = window + pe
#             window=window.squeeze(2)
#             # 构造当前窗口的 I_hat（如果需要针对每个窗口重新构造）
#             if I_hat.shape[1] != M:
#                 I_hat_window = (-1. / M / M) * torch.ones(M, M, device=x.device) + \
#                                (1. / M) * torch.eye(M, M, device=x.device)
#                 I_hat_window = I_hat_window.view(1, M, M).repeat(batch_size, 1, 1).type(x.dtype)
#             else:
#                 I_hat_window = I_hat[:, :M, :M]
#
#             # 计算动态协方差矩阵，与 hb.squeeze(2) @ I_hat @ hb.squeeze(2).mT 形式一致
#             cov_matrix = window @ I_hat_window @ window.mT + torch.eye(c, device=x.device) * 1e-8
#             cov_matrices.append(cov_matrix)  # (batch_size, channels, channels)
#
#         # 汇总协方差矩阵（这里选择平均）
#         dynamic_cov = torch.stack(cov_matrices).mean(dim=0)  # (batch_size, channels, channels)
#         return dynamic_cov

class Vec(nn.Module):
    def __init__(self, input_size):
        super(Vec, self).__init__()
        mask = torch.triu(torch.ones([input_size,input_size], dtype=torch.bool), diagonal=0)
        self.register_buffer('mask', mask)

    def forward(self, input):
        output = input[..., self.mask]
        return output

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        B,C,H,W=x.shape
        return self.pe[:,:W].transpose(-1,-2).unsqueeze(2)


