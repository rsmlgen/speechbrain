from email.policy import strict
from json import load
from locale import normalize
from operator import mod
from turtle import forward
import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.CNN import Conv1d as _Conv1d
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
from speechbrain.nnet.linear import Linear
import speechbrain as sb


class Classifier(sb.nnet.containers.Sequential):
    """This class implements the last MLP on the top of xvector features.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of an example input.
    activation : torch class
        A class for constructing the activation layers.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of output neurons.

    Example
    -------
    >>> input_feats = torch.rand([5, 10, 40])
    >>> compute_xvect = Xvector()
    >>> xvects = compute_xvect(input_feats)
    >>> classify = Classifier(input_shape=xvects.shape)
    >>> output = classify(xvects)
    >>> output.shape
    torch.Size([5, 1, 1211])
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=2048,
        out_neurons=2,
        apply_softmax=False,
    ):
        super().__init__(input_shape=input_shape)

        self.append(activation(), layer_name="act")
        self.append(sb.nnet.normalization.BatchNorm1d, layer_name="norm")

        if lin_blocks > 0:
            self.append(sb.nnet.containers.Sequential, layer_name="DNN")

        for block_index in range(lin_blocks):
            block_name = f"block_{block_index}"
            self.DNN.append(
                sb.nnet.containers.Sequential, layer_name=block_name
            )
            self.DNN[block_name].append(
                sb.nnet.linear.Linear,
                n_neurons=lin_neurons,
                bias=True,
                layer_name="linear",
            )
            self.DNN[block_name].append(activation(), layer_name="act")
            self.DNN[block_name].append(
                sb.nnet.normalization.BatchNorm1d, layer_name="norm"
            )

        # Final Softmax classifier
        self.append(
            sb.nnet.linear.Linear, n_neurons=out_neurons, layer_name="out"
        )
        if apply_softmax:
            self.append(
                sb.nnet.activations.Softmax(apply_log=True), layer_name="softmax"
            )

class LinearBlock(nn.Module):

    def __init__(self, in_feats, out_feats, activation = nn.LeakyReLU(),normalize=True):
        super(LinearBlock, self).__init__()
        self.linear =nn.Linear(in_feats,out_feats)
        torch.nn.init.xavier_uniform(self.linear.weight)
        self.normalize = normalize
        if normalize:
            self.batch_norm = _BatchNorm1d(input_size=out_feats)
        self.activation = activation

    def forward(self,x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.normalize:
            x = self.batch_norm(x)
        return x

class SSL_speaker_enc(nn.Module):

    def __init__(
        self,
        lin_dims = [1024,512,256],
        activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        for i in range(1,len(lin_dims)-1):
            self.blocks.append(LinearBlock(lin_dims[i-1],lin_dims[i],normalize=False))
            # self.blocks.append(LinearBlock(lin_dims[i-1],lin_dims[i],normalize=True))
            
        self.blocks.append(LinearBlock(lin_dims[-2],lin_dims[-1],activation=None,normalize=False))

    def forward(self,x,lens=None):

        for layer in self.blocks:
            try:
                x = layer(x,lenghts=lens)
            except:
                x = layer(x)

        return x


class SSL_speaker_enc_multi(nn.Module):
    def __init__(
        self,
        num_enc=24,
        enc_dim = [1024,512,256],
        activation=nn.LeakyReLU,
        freeze= False,
        pretrained = None,
        device='cpu'
    ):
        super().__init__()

        self.num_enc = num_enc
        self.enc_dim = enc_dim
        self.blocks = nn.ModuleList()
        self.freeze = freeze
        
        for i in range(num_enc):
            self.blocks.append(SSL_speaker_enc(lin_dims = enc_dim))

        if pretrained is not None:
            self.load_pretrained(pretrained)

        if freeze:
            self.blocks.eval()
            for param in self.blocks.parameters():
                param.requires_grad = False

    def forward(self,x):
        output = torch.zeros((x.size()[0],x.size()[1],x.size(2),self.enc_dim[-1]),device=x.device)
        
        for i in range(x.size()[1]):
                output[:,i] = self.blocks[i](x[:,i])
        return output

    def load_pretrained(self,pretrained_path):
        pretrained_state_dict = torch.load(pretrained_path,map_location='cpu')
        modified_state_dict = {}
        for key in self.state_dict().keys():
            if key in pretrained_state_dict.keys():
                modified_state_dict[key] = pretrained_state_dict[key]
        
        self.load_state_dict(modified_state_dict,strict=False)
        # for child in self.children():
        #     for ii in range(len(child)):
        #         for jj in range(len(child[ii]._modules['blocks'])):
        #             if type(child[ii]._modules['blocks'][jj])== _BatchNorm1d:
        #                 child[ii].track_running_stats = False
        # print()

    

class WeightedSum(nn.Module):
    def __init__(
        self,
        num_weight=24,
        freeze = False,
        pretrained = None,
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.rand((num_weight,1)))
        self.freeze = freeze
        
        if pretrained is not None:
            self.load_pretrained(pretrained)
        
        if freeze:
            self.weights.requires_grad = False

    def forward(self,x):
        
        # x_t = torch.transpose(x,0,-1)
        x_t = torch.permute(x,(-1,0,2,1))
        w_sum = torch.matmul(x_t,self.weights) / torch.sum(self.weights)

        return w_sum.squeeze(-1).permute(1,2,0)

    def load_pretrained(self,pretrain_path):
        pretrained_state_dict = torch.load(pretrain_path,map_location='cpu')
        self.load_state_dict(pretrained_state_dict,strict=False)


class WeightedSum_softmax(nn.Module):
    def __init__(
        self,
        num_weight=24,
        freeze = False,
        pretrained = None,
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros((num_weight,1)))
        self.freeze = freeze
        
        if pretrained is not None:
            self.load_pretrained(pretrained)
        
        if freeze:
            self.weights.requires_grad = False

    def forward(self,x):
        
        # x_t = torch.transpose(x,0,-1)
        x_t = torch.permute(x,(-1,0,2,1))
        weights_norm = F.softmax(self.weights,dim=0)
        w_sum = torch.matmul(x_t,weights_norm)

        return w_sum.squeeze(-1).permute(1,2,0)

    def load_pretrained(self,pretrain_path):
        pretrained_state_dict = torch.load(pretrain_path,map_location='cpu')
        self.load_state_dict(pretrained_state_dict,strict=False)



class AvgPooling(nn.Module):
    def __init__(
        self,
        use_std = False,
    ):
        super().__init__() 
        self.use_std = use_std 


    def forward(self,x):
        
        x_mean = torch.mean(x,dim=-2)
        if self.use_std:
            x_std = torch.std(x,dim=-2)

            x_pooled = torch.cat(x_mean,x_std,dim=-2)
        else:
            x_pooled = x_mean

        return x_pooled

class AvgPooling_test(nn.Module):
    def __init__(
        self,
        use_std = False,
    ):
        super().__init__() 
        self.use_std = use_std 


    def forward(self,x,lens):
        
        
        x_pooled = torch.zeros((x.size()[0],x.size()[2]),dtype=x.dtype,device=x.device)

        for i in range(x.size()[0]):
            actual_size = torch.floor(lens[i]*x.size()[1]).int()
            x_pooled[i] = torch.mean(x[i][:actual_size],dim=-2)

        return x_pooled

