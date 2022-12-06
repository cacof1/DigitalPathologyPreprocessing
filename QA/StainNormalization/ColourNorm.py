import torch.nn as nn
import torch

class Macenko(nn.Module):
    # Macenko colour normalisation. Takes as input a torch tensor (intensity ranging [0,255])
    # alpha: percentile for normalisation (considers data within alpha and (100-alpha) percentiles).
    # beta: threshold of normalisation values for analysis.
    # Io: (optional) transmitted light intensity
    def __init__(self, alpha=0.01, beta=0.65, Io=255, saved_fit_file=None, get_stains=False, normalize=True, HE= None):
        super(Macenko, self).__init__()
        self.alpha = alpha 
        self.beta = beta
        self.get_stains = get_stains
        self.Io = Io
        self.normalize = normalize
        # Default fit values reported in the original git code (origin unclear)
        self.HERef = torch.tensor([[0.5626, 0.2159],
                                   [0.7201, 0.8012],
                                   [0.4062, 0.5581]])
        self.maxCRef = torch.tensor([1.9705, 1.0308])

    def forward(self, img, fit=None): # Expect H x W x C

        h, w, c    = img.shape
        img        = img.reshape(-1, img.shape[-1]) ## Collapse H x W -> N
        OD, ODhat  = self.convert_rgb2od(img)
        if ODhat.shape[0] <= 10:  # this slide is bad for processing - too many transparent points.
            img_norm = img.reshape(h, w, c).int()
            if(self.get_stains): return img, None, None, None, self.HERef
            else: return img

        HE        = self.find_HE(ODhat) 
        C         = torch.linalg.lstsq(HE, OD.T)[0]  # determine concentrations of the individual stains
        maxC      = torch.stack([torch.quantile(C[0, :], 0.99), torch.quantile(C[1, :], 0.99)])
        if(self.normalize): C *= (self.maxCRef / maxC).unsqueeze(-1)  # normalize stain concentrations

        # recreate the image using reference stain vectors
        img_norm = self.Io * torch.exp(-torch.matmul(self.HERef, C))
        img_norm[img_norm > 255] = 255
        img_norm = img_norm.T.reshape(h, w, c).int()
        if self.get_stains:
            H = torch.mul(self.Io, torch.exp(torch.matmul(-self.HERef[:, 0].unsqueeze(-1), C[0, :].unsqueeze(0))))
            H[H > 255] = 255
            H = H.T.reshape(h, w, c).int()

            E = torch.mul(self.Io, torch.exp(torch.matmul(-self.HERef[:, 1].unsqueeze(-1), C[1, :].unsqueeze(0))))
            E[E > 255] = 255
            E = E.T.reshape(h, w, c).int()
            return img_norm, H, E, HE, maxC
        else:
            return img_norm                

    def convert_rgb2od(self, img):
        OD    = -torch.log((img.float())/ self.Io)
        ODhat = OD[torch.sum(OD,dim=1) > self.beta]  # remove transparent pixels
        return OD, ODhat

    def find_HE(self, ODhat):
        eigvalues, eigvecs = torch.linalg.eigh(torch.cov(ODhat.T), UPLO='L')
        eigvecs            = eigvecs[:, [1, 2]]
        
        # project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        That   = torch.matmul(ODhat, eigvecs)
        phi    = torch.atan2(That[:, 1], That[:, 0])
        print(That,phi, ODhat)
        minPhi = torch.quantile(phi, self.alpha)
        maxPhi = torch.quantile(phi, 1 - self.alpha)
        vMin   = torch.matmul(eigvecs, torch.stack((torch.cos(minPhi), torch.sin(minPhi)))).unsqueeze(1)
        vMax   = torch.matmul(eigvecs, torch.stack((torch.cos(maxPhi), torch.sin(maxPhi)))).unsqueeze(1)

        # a heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second
        HE = torch.where(vMin[0] > vMax[0], torch.cat((vMin, vMax), dim=1), torch.cat((vMax, vMin), dim=1))
        return HE
