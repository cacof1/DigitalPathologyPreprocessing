import numpy as np
import openslide
from matplotlib import pyplot as plt
import os
import torch
import ColourNorm
import staintools
from skimage.transform import resize
from sys import platform
import seaborn as sns
import matplotlib
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D


# Select training slide
id_train = '484813'
bp0 = ""
bp = "./Slides"

# Select testing slide
# Some slides we have tested: RNOH_S00104604_164604, RNOH_S00104812_165047, RNOH_S00104628_164811, 492090, 492007, 492040, 499877, 493199.
id_test = 'RNOH_S00104812_165047'

# --------------------------------------------------------------------------------------------------------------------
# Fixed parameters of Macenko normalisation - see class for more details.
Io = 255
export = False

# Visibility level for processing. LEAVE TO ZERO, otherwise colour norm will be biased to a zoom level.
vis = 0

# Load training image and select adequate patch of image.
START = (50000, 31000)
SIZE  = (500, 500)
wsi_object_train = openslide.open_slide(os.path.join(bp0, bp, id_train + '.svs'))
img_train = np.array(wsi_object_train.read_region(START, vis, SIZE).convert("RGB"))

# Train
print('Starting to train....')
MacenkoNormaliser  = ColourNorm.Macenko(get_stains=True, normalize=False)
#img_train_for_norm = torch.from_numpy(img_train).permute(2, 0, 1)  # torch of size C x H x W.
#MacenkoNormaliser.fit(img_train_for_norm)
print('Training completed.')

# Also train with staintools (other normaliser from literature, to compare with our implementation)
print('Starting to train with staintools....')
MacenkoNormaliser_Staintools = staintools.StainNormalizer(method="Macenko")
MacenkoNormaliser_Staintools.fit(img_train)
print('Training completed.')
img_train = resize(img_train, (512, 512))  # reshape for display because the training ROI is too large.
# --------------------------------------------------------------------------------------------------------------------
# Fit multiple test slides

#wsi_object_test = openslide.open_slide(os.path.join(bp0, bp, id_test + '.svs'))

# Multiple test slides
id_tests = ['RNOH__123138','RNOH__124701','500135','500135','500006','492059','RNOH_S00125416_133727','RNOH_S00104812_165047']
starts = [(53850, 57900), (72000, 52000), (47000, 46000), (6000, 19000), (56500, 54500), (48000, 53000), (45000, 54000), (26000, 41000)]
sizes = [(2048, 2048)]*len(starts) ##2048
maxC_Array = []
for n, (start, size, id_test) in enumerate(zip(starts, sizes, id_tests)): 
#for start, size in zip(starts, sizes):
    if(n<=1): continue
    wsi_object_test = openslide.open_slide(os.path.join(bp0, bp, id_test + '.svs'))
    img_test_for_HE = np.array(wsi_object_test.read_region(START, vis, SIZE).convert("RGB"))
    print('Processing ROI ({},{})...'.format(start[0], start[1]))
    img_test = np.array(wsi_object_test.read_region(start, vis, size).convert("RGB"))

    print(img_test.shape)
    # Fit - comparison with staintools
    img_test_norm_Staintools = MacenkoNormaliser_Staintools.transform(img_test)
    
    # Fit - our implementation
    img_test_norm, H, E, HE,  maxC   = MacenkoNormaliser.forward(torch.from_numpy(img_test))
    img_test_norm                    = img_test_norm.cpu().detach().numpy()
    maxC_Array.append(maxC.cpu().detach().numpy())

    maxC = maxC.cpu().detach().numpy()

    plt.figure(figsize=[16,8])
    plt.subplot(2, 4, 1)
    plt.imshow(resize(img_train, (512, 512)))  # reshape for display because the training ROI is too large.
    plt.title('Reference image')

    plt.subplot(2, 4, 2)
    plt.imshow(img_test)
    plt.title('Original test image')

    plt.subplot(2, 4, 3)
    plt.imshow(img_test_norm)
    plt.title('Macenko - Norm test image')

    #plt.subplot(2, 4, 4)
    #plt.imshow(img_test_nonorm)
    #plt.title('Macenko - No Norm test image')    

    plt.subplot(2, 4, 5)
    plt.imshow(H)
    plt.title('Haematoxylin channel')

    plt.subplot(2, 4, 6)
    plt.imshow(E)
    plt.title('Eosin channel')

    plt.subplot(2, 4, 7)
    plt.imshow(img_test_norm_Staintools)
    plt.title('Normalized with staintools')
    #plt.savefig("Macenko_LargeRef_LargeTest_"+str(id_test)+".png")
    plt.show()


    # Histograms
    #Rn, Gn, Bn = img_test_nonorm[:,:,0].flatten(), img_test_nonorm[:,:,1].flatten(), img_test_nonorm[:,:,2].flatten()
    #R, G, B    = img_test_norm[:, :, 0].flatten(), img_test_norm[:, :, 1].flatten(), img_test_norm[:, :, 2].flatten()

    """    
    bins = np.linspace(0, 1, 30)
    Ap = 0.7
    dens = True
    fig, ax = plt.subplot_mosaic("ABC", figsize=(17, 4))    
    ax['A'].hist(R, bins=bins, alpha=Ap, color='darkred', density=dens)
    ax['A'].hist(Rn, bins=bins, alpha=Ap, color='lightcoral', density=dens)
    ax['A'].legend(['Raw','Norm'])
    ax['B'].set_title('R')
    
    ax['B'].hist(G, bins=bins, alpha=Ap, color='darkgreen', density=dens)
    ax['B'].hist(Gn, bins=bins, alpha=Ap, color='springgreen', density=dens)
    ax['B'].legend(['Raw','Norm'])
    ax['B'].set_title('G')
    
    ax['C'].hist(B, bins=bins, alpha=Ap, color='darkblue', density=dens)
    ax['C'].hist(Bn, bins=bins, alpha=Ap, color='cornflowerblue', density=dens)
    ax['C'].legend(['Raw','Norm'])
    ax['C'].set_title('B')
    plt.show()
    """
    """
    cs= np.array([[float(R_)/255,float(G_)/255,float(B_)/255,1] for R_,G_,B_ in zip(Rn,Gn,Bn)])
    cs2= np.array([[float(R_)/255,float(G_)/255,float(B_)/255,1] for R_,G_,B_ in zip(R,G,B)])
    
    img_test      =  -np.log10((img_test.astype("float32") + 1) / 255)
    img_test_norm =  -np.log10((img_test_norm.astype("float32") + 1) / 255)    
    Rn, Gn, Bn = img_test[:,:,0].flatten(), img_test[:,:,1].flatten(), img_test[:,:,2].flatten()
    R, G, B    = img_test_norm[:, :, 0].flatten(), img_test_norm[:, :, 1].flatten(), img_test_norm[:, :, 2].flatten()


    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')
    #ax1.scatter(Rn, Gn, Bn, c= cs)
    ax1.plot([0,HE[0,0]], [0,HE[1,0]], [0,HE[2,0]], c='b')
    ax1.plot([0,HE[0,1]], [0,HE[1,1]], [0,HE[2,1]], c='r')
    #ax1.plot([0,HERef[0,0]], [0,HERef[1,0]], [0,HERef[2,0]],'b--')
    #ax1.plot([0,HERef[0,1]], [0,HERef[1,1]], [0,HERef[2,1]],'r--')        
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    #ax2.plot([0,HE[0,0]], [0,HE[1,0]], [0,HE[2,0]], c='b')
    #ax2.plot([0,HE[0,1]], [0,HE[1,1]], [0,HE[2,1]], c='r')
    ax2.plot([0,HERef[0,0]], [0,HERef[1,0]], [0,HERef[2,0]],'b--')
    ax2.plot([0,HERef[0,1]], [0,HERef[1,1]], [0,HERef[2,1]],'r--')            
    ax2.scatter(R, G, B, c = cs2,alpha=0.5)    
    plt.show()
    """

maxC_Array = np.array(maxC_Array)
print(maxC_Array)
fig, ax = plt.subplots()
ax.plot(maxC_Array[:,0],'ro',label='Eosin Concentration')
ax.plot(maxC_Array[:,1],'bo',label='Hematoxylin Concentraion')
print(maxC_Array.shape, len(id_tests))
ax.set_xticklabels([''] +id_tests,rotation=45)
ax.set_ylabel('Optical Density')
fig.tight_layout()
plt.legend(frameon=False)
plt.show()
# Export the current colour calibration.
if export:

    filename = './trained/' + id_train + '_vis' + str(vis) + '_HERef.pt'
    HEref = MacenkoNormaliser.HERef
    maxCRef = MacenkoNormaliser.maxCRef
    alpha = MacenkoNormaliser.alpha
    beta = MacenkoNormaliser.beta
    Io = MacenkoNormaliser.Io
    torch.save({'HERef': HEref, 'maxCRef': maxCRef, 'alpha': alpha, 'beta': beta, 'Io': Io}, filename)


    
