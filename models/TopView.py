import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_

def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )

def predict_top_view(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )

def crop_like(input, ref_size2, ref_size3):
    assert(input.size(2) >= ref_size2 and input.size(3) >= ref_size3)
    return input[:, :, :ref_size2, :ref_size3]


class TopView(nn.Module):

    def __init__(self, ref_size2, ref_size3, last_depth):
        super(TopView, self).__init__()

        self.ref_size2 = ref_size2
        self.ref_size3 = ref_size3

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]

        assert(len(upconv_planes) == len(ref_size2) and len(upconv_planes) == len(ref_size3))        

        self.upconv7 = upconv(last_depth,   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.pred_tv = predict_top_view(upconv_planes[6])


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
                    

    def forward(self, x):

    	out_upconv7 = crop_like(upconv7(x), ref_size2[6], ref_size3[6]) 
    	out_upconv6 = crop_like(upconv6(out_upconv7), ref_size2[5], ref_size3[5])
    	out_upconv5 = crop_like(upconv5(out_upconv6), ref_size2[4], ref_size3[4])
    	out_upconv4 = crop_like(upconv4(out_upconv5), ref_size2[3], ref_size3[3])
    	out_upconv3 = crop_like(upconv3(out_upconv4), ref_size2[2], ref_size3[2])
    	out_upconv2 = crop_like(upconv2(out_upconv3), ref_size2[1], ref_size3[1])
    	out_upconv1 = crop_like(upconv1(out_upconv2), ref_size2[0], ref_size3[0])

    	top_view = pred_tv(upconv1)

    	return top_view



