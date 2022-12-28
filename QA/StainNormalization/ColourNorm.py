import torch.nn as nn
import torch
from typing import Union


class Macenko(nn.Module):
    # Macenko colour normalisation. Takes as input a torch tensor (intensity ranging [0,255])
    # alpha: percentile for normalisation (considers data within alpha and (100-alpha) percentiles).
    # beta: threshold of normalisation values for analysis.
    # Io: (optional) transmitted light intensity
    def __init__(self, alpha=1, beta=0.45, Io=255, get_stains=False, normalize=True, HERef=None, maxCRef=None):
        super(Macenko, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.get_stains = get_stains
        self.Io = Io
        self.normalize = normalize
        # Default fit values reported in the original git code (origin unclear)

        if HERef is None:
            self.HERef = torch.tensor([[0.5626, 0.2159],
                                       [0.7201, 0.8012],
                                       [0.4062, 0.5581]])
        else:
            self.HERef = torch.tensor(HERef)

        if maxCRef is None:
            self.maxCRef = torch.tensor([1.9705, 1.0308])
        else:
            self.maxCRef = torch.tensor(maxCRef)

    def forward(self, img, HE=None, maxC=None):  # img has expected size C x H x W
        c, h, w = img.shape
        img = img.reshape(img.shape[0], -1)  # Collapse H x W -> N
        OD, ODhat = self.convert_rgb2od(img)

        if ODhat.shape[1] <= 10:  # this slide is bad for processing - too many transparent points.
            img = img.reshape(c, h, w)
            if self.get_stains:
                return img, None, None, None, self.HERef
            else:
                return img

        C = torch.linalg.lstsq(HE, OD)[0]  # determine concentrations of the individual stains
        if maxC is None:  # otherwise use the input.
            maxC = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])

        if (self.normalize): C *= (self.maxCRef / maxC).unsqueeze(-1)  # normalize stain concentrations

        # recreate the image using reference stain vectors
        img_norm = torch.exp(-torch.matmul(self.HERef, C))
        img_norm = img_norm.reshape(c, h, w)

        if self.get_stains:
            H = torch.exp(torch.matmul(-self.HERef[:, 0].unsqueeze(-1), C[0, :].unsqueeze(0)))
            H = H.T.reshape(c, h, w)

            E = torch.exp(torch.matmul(-self.HERef[:, 1].unsqueeze(-1), C[1, :].unsqueeze(0)))
            E = E.T.reshape(c, h, w)
            return img_norm, H, E, self.HE, maxC
        else:
            return img_norm

    def convert_rgb2od(self, img):  # img has expected size C x H x W
        OD = -torch.log(img)
        ODhat = OD[:, torch.sum(OD, dim=0) > self.beta]  # remove transparent pixels
        ODhat = ODhat[:, ~torch.any(ODhat == torch.inf, dim=0)]  # fail-safe for any pixels which had intensity = 0.
        return OD, ODhat

    def find_HE(self, img, get_maxC=False):  # img has expected size C x H x W
        img = img.reshape(img.shape[0], -1)  # Collapse H x W -> N
        
        OD, ODhat = self.convert_rgb2od(img)
        del img
        if ODhat.shape[1] <= 10: return self.HERef, self.maxCRef  # this slide is bad for processing - too many transparent points.

        eigvalues, eigvecs = torch.linalg.eigh(torch.cov(ODhat), UPLO='L')
        eigvecs = eigvecs[:, [1, 2]]

        # project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        That = torch.matmul(ODhat.T, eigvecs)
        phi = torch.atan2(That[:, 1], That[:, 0])
        minPhi = percentile(phi, self.alpha)
        maxPhi = percentile(phi, 100 - self.alpha)
        vMin = torch.matmul(eigvecs, torch.stack((torch.cos(minPhi), torch.sin(minPhi)))).unsqueeze(1)
        vMax = torch.matmul(eigvecs, torch.stack((torch.cos(maxPhi), torch.sin(maxPhi)))).unsqueeze(1)
        del ODhat
        # a heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second
        HE = torch.where(vMin[0] > vMax[0], torch.cat((vMin, vMax), dim=1), torch.cat((vMax, vMin), dim=1))

        if get_maxC:
            C = torch.linalg.lstsq(HE, OD)[0]  # determine concentrations of the individual stains
            maxC = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])
            return HE, maxC
        else:
            return HE


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    k = 1 + round(.01 * float(q) * (t.numel() - 1))

    if k == 0 or t.size()[0] == 0:
        out = torch.tensor(0)  # default to dummy value if there is no point of interest in slide. Will not affect
        # the results, this is just a fail safe.
    else:
        out = t.view(-1).kthvalue(int(k)).values

    return out
