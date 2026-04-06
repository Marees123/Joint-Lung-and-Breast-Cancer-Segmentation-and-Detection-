import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def Image_Results():
    I = [[11, 16, 21, 36, 81], [2, 5, 8, 12, 32]]
    for n in range(2):
        Images = np.load('Images_'+str(n+1)+'.npy', allow_pickle=True)
        GT = np.load('GT_'+str(n+1)+'.npy', allow_pickle=True)
        UNET = np.load('Unet_'+str(n+1)+'.npy', allow_pickle=True)
        Resunet = np.load('Resunet_'+str(n+1)+'.npy', allow_pickle=True)
        DD_Attention_ResUNet = np.load('DD_Attention_ResUNet_'+str(n+1)+'.npy', allow_pickle=True)
        DA_ViT_UNetPP = np.load('DA_ViT_UNetPP_'+str(n+1)+'.npy', allow_pickle=True)
        Proposed = np.load('Proposed_'+str(n+1)+'.npy', allow_pickle=True)
        for i in range(len(I[n])):
            plt.subplot(2, 3, 1)
            plt.title('Original')
            plt.imshow(Images[I[n][i]])
            plt.subplot(2, 3, 2)
            plt.title('GroundTruth')
            plt.imshow(GT[I[n][i]])
            plt.subplot(2, 3, 3)
            plt.title('UNET')
            plt.imshow(UNET[I[n][i]])
            plt.subplot(2, 3, 4)
            plt.title('Resunet')
            plt.imshow(Resunet[I[n][i]])
            plt.subplot(2, 3, 5)
            plt.title('DA_ViT_UNetPP')
            plt.imshow(DA_ViT_UNetPP[I[n][i]])
            plt.subplot(2, 3, 6)
            plt.title('Proposed')
            plt.imshow(Proposed[I[n][i]])
            plt.tight_layout()
            plt.show()
            # cv.imwrite('./Results/Image_Results/' + str(n + 1) + 'orig-' + str(i + 1) + '.png', Images[I[0][i]])
            # cv.imwrite('./Results/Image_Results/' + str(n + 1) + 'gt-' + str(i + 1) + '.png', GT[I[0][i]])
            # cv.imwrite('./Results/Image_Results/' + str(n + 1) + 'ResUnet-' + str(i + 1) + '.png', UNET[I[0][i]])
            # cv.imwrite('./Results/Image_Results/' + str(n + 1) + 'DDA-AttResUNet-' + str(i + 1) + '.png',
            #            DD_Attention_ResUNet[I[0][i]])
            # cv.imwrite('./Results/Image_Results/' + str(n + 1) + 'AEDVNS-EUNet++-' + str(i + 1) + '.png',
            #            DA_ViT_UNetPP[I[0][i]])
            # cv.imwrite('./Results/Image_Results/' + str(n + 1) + 'proposed-' + str(i + 1) + '.png',
            #            Proposed[I[0][i]])


def Sample_Images():
    for n in range(2):
        Orig = np.load('Images_'+str(n+1)+'.npy', allow_pickle=True)
        ind = [126, 127, 128, 129, 140, 141]
        fig, ax = plt.subplots(2, 3)
        plt.suptitle("Sample Images from Dataset ")
        plt.subplot(2, 3, 1)
        plt.title('Image-1')
        plt.imshow(Orig[ind[0]])
        plt.subplot(2, 3, 2)
        plt.title('Image-2')
        plt.imshow(Orig[ind[1]])
        plt.subplot(2, 3, 3)
        plt.title('Image-3')
        plt.imshow(Orig[ind[2]])
        plt.subplot(2, 3, 4)
        plt.title('Image-4')
        plt.imshow(Orig[ind[3]])
        plt.subplot(2, 3, 5)
        plt.title('Image-5')
        plt.imshow(Orig[ind[4]])
        # plt.show()
        plt.subplot(2, 3, 6)
        plt.title('Image-6')
        plt.imshow(Orig[ind[5]])
        plt.show()


if __name__ == '__main__':
    Image_Results()
    Sample_Images()
