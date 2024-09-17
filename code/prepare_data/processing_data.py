import os
import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage as nd
from scipy import ndimage
from skimage import filters
from skimage import io
import torch
import torch.fft
from build_geometry import initialization, build_geometry
from matplotlib import pyplot as plt

CTNVIEW=64
MRIDOWN=4
param = initialization()
param.param['nProj'] = CTNVIEW
reco_space, ray_trafo, FBPOper, _ = build_geometry(param)


def simulate_sinogram(imgCT):
    img = imgCT / 1000 * 0.192 + 0.192
    img[img < 0.0] = 0.0
    sinogram = ray_trafo(img)

    fbp_ct = np.asarray(FBPOper(sinogram))
    fbp_ct[fbp_ct < 0.0] = 0.0

    return sinogram, fbp_ct


class MaskFunc_Cartesian:
    """
    MaskFunc creates a sub-sampling mask of a given shape.
    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        a) N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        b) The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)
    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.
    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.
    """

    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.
            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)
        num_cols = shape[-2]

        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs + 1e-10)
        mask = self.rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        mask = mask.repeat(shape[0], 1, 1)

        return mask

## mri related
def mri_fourier_transform_2d(image, mask):
    '''
  image: input tensor [B, H, W, C]
  mask: mask tensor [H, W]
  '''
    spectrum = torch.fft.fftn(image, dim=(1, 2))
    # K-space spectrum has been shifted to shift the zero-frequency component to the center of the spectrum
    spectrum = torch.fft.fftshift(spectrum, dim=(1, 2))
    # Downsample k-space
    spectrum = spectrum * mask[None, :, :, None]
    return spectrum

## mri related
def mri_inver_fourier_transform_2d(spectrum):
    '''
  image: input tensor [B, H, W, C]
  '''
    spectrum = torch.fft.ifftshift(spectrum, dim=(1, 2))
    image = torch.fft.ifftn(spectrum, dim=(1, 2))

    return image

def simulate_undersample_mri(raw_mri):
    mri = torch.tensor(raw_mri)[None, :, :, None]
    ff = MaskFunc_Cartesian([0.1], [MRIDOWN])
    shape = [384, 384, 1]
    mask = ff(shape, seed=1337)
    mask = mask[:, :, 0]

    kspace = mri_fourier_transform_2d(mri, mask)
    mri_recon = mri_inver_fourier_transform_2d(kspace)
    kdata = torch.sqrt(kspace.real ** 2 + kspace.imag ** 2 + 1e-10)
    kdata = kdata.data.numpy()[0, :, :, 0]

    under_img = torch.sqrt(mri_recon.real ** 2 + mri_recon.imag ** 2)
    under_img = under_img.data.numpy()[0, :, :, 0]

    return under_img, kspace

def _parse(rootdir):
    filenames = [f for f in os.listdir(rootdir) if f.endswith('.nii')]
    filenames.sort()
    filetree = {}

    for filename in filenames:
        subject, modality = filename.split('.').pop(0).split('_')[:2]

        if subject not in filetree:
            filetree[subject] = {}
        filetree[subject][modality] = filename

    return filetree


def clean(rootdir, source_modality, target_modality):
    filetree = _parse(rootdir+'/raw_data')

    if not os.path.exists(rootdir+'/img_norm'):
        os.makedirs(rootdir+'/img_norm')
    
    if not os.path.exists(rootdir+'/img_test'):
        os.makedirs(rootdir+'/img_test')

    for subject, modalities in filetree.items():
        print(f'{subject}:')

        if source_modality not in modalities or target_modality not in modalities:
            print('-> incomplete')
            continue

        source_path = os.path.join(rootdir, 'raw_data', modalities[source_modality])
        target_path = os.path.join(rootdir, 'raw_data', modalities[target_modality])

        source_image = nib.load(source_path)
        target_image = nib.load(target_path)

        source_volume = source_image.get_fdata()
        target_volume = target_image.get_fdata()
        binary_volume = np.zeros_like(source_volume)

        for i in range(binary_volume.shape[-1]):
            source_slice = source_volume[:, :, i]

            if source_slice.min() == source_slice.max():
                binary_volume[:, :, i] = np.zeros_like(source_slice)
            else:
                threshold = filters.threshold_li(source_slice)

                binary_volume[:, :, i] = ndimage.morphology.binary_fill_holes(
                    source_slice > threshold)

        source_volume = np.where(binary_volume, source_volume, np.ones_like(
            source_volume) * source_volume.min())
        target_volume = np.where(binary_volume, target_volume, np.ones_like(
            target_volume) * target_volume.min())
        ## resize
        if source_image.header.get_zooms()[0] < 0.6:
            scale = np.asarray([384, 384, source_volume.shape[-1]]) / np.asarray(source_volume.shape)
            source_volume = nd.zoom(source_volume, zoom=scale, order=3, prefilter=False)
            target_volume = nd.zoom(target_volume, zoom=scale, order=0, prefilter=False)

        # save volume into images
        source_volume = (source_volume-source_volume.min())/(source_volume.max()-source_volume.min())

        for i in range(binary_volume.shape[-1]):
            binary_slice = binary_volume[:, :, i]
            if binary_slice.max() > 0:
                dd = target_volume.shape[0] // 2
                target_slice = target_volume[dd - 192:dd + 192, dd - 192:dd + 192, i]
                source_slice = source_volume[dd - 192:dd + 192, dd - 192:dd + 192, i]
                # sparse CT and undersample MRI
                sinogram, fbp_ct = simulate_sinogram(target_slice)
                under_img, kspace = simulate_undersample_mri(source_slice)

                io.imsave(rootdir+'/img_norm/'+subject+'_'+str(i)+'_'+source_modality+'.png', source_slice)
                tmp_target_slice = target_slice/1000 * 0.192 + 0.192
                tmp_target_slice[tmp_target_slice < 0.0] = 0.0
                io.imsave(rootdir+'/img_norm/'+subject+'_'+str(i)+'_'+target_modality+'.png', tmp_target_slice)
                io.imsave(rootdir + '/img_norm/' + subject + '_' + str(i) + '_' + target_modality + '_fbpct.png',
                          (fbp_ct * 255).astype(np.uint8))
                io.imsave(rootdir + '/img_norm/' + subject + '_' + str(i) + '_' + target_modality + '_undermri.png',
                          (under_img * 255).astype(np.uint8))
                np.savez_compressed(rootdir + '/img_norm/' + subject + '_' + str(i) + '_' + target_modality + '_raw_'+str(MRIDOWN)+'X'+str(CTNVIEW)+'P',
                                    sino=sinogram, fbp=fbp_ct,
                                    kspace=kspace, under_t1=under_img,
                                    t1=source_slice, ct=target_slice)


def main(args):
    clean(args.rootdir, args.source, args.target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rootdir')
    parser.add_argument('--source', default='t1')
    parser.add_argument('--target', default='ct')

    main(parser.parse_args())
