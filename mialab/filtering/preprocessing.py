"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def show_image(image, title='Image', cmap='gray'):
    """
    Function to display an MRI image using Matplotlib.
    Args:
        image (SimpleITK.Image): The image to display.
        title (str): Title of the plot.
        cmap (str): Color map to use for the image display.
    """
    # Convert the SimpleITK image to a numpy array for visualization
    image_array = sitk.GetArrayFromImage(image)

    # Display the image (assumes 3D or 2D image)
    plt.figure(figsize=(6, 6))
    plt.imshow(image_array[image_array.shape[0] // 2], cmap=cmap)  # Show the middle slice of the 3D image
    plt.title(title)
    plt.axis('off')
    plt.show()

class ImageNormalization(pymia_fltr.Filter):
    """Represents a normalization filter."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.
        """

        img_arr = sitk.GetArrayFromImage(image)

        # todo: normalize the image using numpy
        #warnings.warn('No normalization implemented. Returning unprocessed image.')

        if np.max(img_arr) == np.min(img_arr):
            return image

        # Normalize the image: scale pixel values to [0, 1]
        img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageNormalization:\n' \
            .format(self=self)
    
class ImageResampling(pymia_fltr.Filter):
    """Represents an image resampling filter."""

    def __init__(self):
        """Initializes a new instance of the ImageResampling class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams) -> sitk.Image:
        """Executes a resampling on an image.

        Args:
            image (sitk.Image): The image to be resampled.
            params (pymia_fltr.FilterParams): The resampling parameters, including
                'new_size' (tuple): The new size of the image (width, height).
                'interpolation' (str): The interpolation type ('bilinear', 'spline', 'nearest').

        Returns:
            sitk.Image: The resampled image.
        """
        if params is None or 'new_size' not in params or 'interpolation' not in params:
            raise ValueError("Parameters must include 'new_size' and 'interpolation'.")

        new_size = params['new_size']
        interpolation = params['interpolation']

        # Determine the interpolator
        if interpolation == 'bilinear':
            interpolator = sitk.sitkLinear
        elif interpolation == 'spline':
            interpolator = sitk.sitkBSpline
        elif interpolation == 'nearest':
            interpolator = sitk.sitkNearestNeighbor
        else:
            raise ValueError(f"Unsupported interpolation type: {interpolation}")
        print('[resampling method]')
        # Perform resampling
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing([os / ns * s for os, ns, s in zip(image.GetSize(), new_size, image.GetSpacing())])
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())

        return resampler.Execute(image)

class ImageDenoising(pymia_fltr.Filter):
    """Represents an image denoising filter."""

    def __init__(self):
        """Initializes a new instance of the ImageDenoising class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes a denoising operation on an image.

        Args:
            image (sitk.Image): The image to be denoised.
            params (FilterParams): The parameters, including:
                'method' (str): The denoising method ('wiener', 'gaussian', 'median').
                Additional parameters depending on the method.

        Returns:
            sitk.Image: The denoised image.
        """
        if params is None or 'method' not in params:
            raise ValueError("Parameters must include 'method'.")

        method = params['method']

        if method == 'wiener':
            return self._apply_wiener_filter(image, params)
        elif method == 'gaussian':
            return self._apply_gaussian_smoothing(image, params)
        elif method == 'median':
            return self._apply_median_filtering(image, params)
        else:
            raise ValueError(f"Unsupported denoising method: {method}")

    def _apply_wiener_filter(self, image: sitk.Image, params: pymia_fltr.FilterParams) -> sitk.Image:
        """Applies a Wiener filter to the image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters (optional).

        Returns:
            sitk.Image: The denoised image.
        """
        # Wiener filter isn't directly implemented in SimpleITK, but we can approximate it
        img_arr = sitk.GetArrayFromImage(image)
        # Approximate Wiener filter using scipy (for demonstration)
        import scipy.signal
        kernel_size = params.get('kernel_size', 3)
        img_denoised = scipy.signal.wiener(img_arr, kernel_size)

        img_out = sitk.GetImageFromArray(img_denoised)
        img_out.CopyInformation(image)
        return img_out

    def _apply_gaussian_smoothing(self, image: sitk.Image, params: pymia_fltr.FilterParams) -> sitk.Image:
        """Applies Gaussian smoothing to the image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters, including:
                'sigma' (float): The Gaussian standard deviation.

        Returns:
            sitk.Image: The smoothed image.
        """
        sigma = params.get('sigma', 1.0)
        return sitk.SmoothingRecursiveGaussian(image, sigma)

    def _apply_median_filtering(self, image: sitk.Image, params: pymia_fltr.FilterParams) -> sitk.Image:
        """Applies median filtering to the image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters, including:
                'radius' (int): The radius of the median filter.

        Returns:
            sitk.Image: The filtered image.
        """
        radius = params.get('radius', 1)
        return sitk.Median(image, [radius] * image.GetDimension())


class SkullStrippingParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the SkullStrippingParameters

        Args:
            img_mask (sitk.Image): The brain mask image.
        """
        self.img_mask = img_mask


class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: SkullStrippingParameters = None) -> sitk.Image:
        """Executes a skull stripping on an image.

        Args:
            image (sitk.Image): The image.
            params (SkullStrippingParameters): The parameters with the brain mask.

        Returns:
            sitk.Image: The normalized image.
        """
        mask = params.img_mask  # the brain mask

        # todo: remove the skull from the image by using the brain mask
        #warnings.warn('No skull-stripping implemented. Returning unprocessed image.')

        image = sitk.Mask(image, mask)
        #show_image(image, title='Image after skull stripping')
        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'SkullStripping:\n' \
            .format(self=self)


class ImageRegistrationParameters(pymia_fltr.FilterParams):
    """Image registration parameters."""

    def __init__(self, atlas: sitk.Image, transformation: sitk.Transform, is_ground_truth: bool = False):
        """Initializes a new instance of the ImageRegistrationParameters

        Args:
            atlas (sitk.Image): The atlas image.
            transformation (sitk.Transform): The transformation for registration.
            is_ground_truth (bool): Indicates weather the registration is performed on the ground truth or not.
        """
        self.atlas = atlas
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth


class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: ImageRegistrationParameters = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (ImageRegistrationParameters): The registration parameters.

        Returns:
            sitk.Image: The registered image.
        """

        # todo: replace this filter by a registration. Registration can be costly, therefore, we provide you the
        # transformation, which you only need to apply to the image!
        #warnings.warn('No registration implemented. Returning unregistered image')

        atlas = params.atlas
        transform = params.transformation
        is_ground_truth = params.is_ground_truth  # the ground truth will be handled slightly different

        image = sitk.Resample(image, atlas, transform, sitk.sitkLinear, 0.0)
        show_image(image, title='Image after registration')

        # note: if you are interested in registration, and want to test it, have a look at
        # pymia.filtering.registration.MultiModalRegistration. Think about the type of registration, i.e.
        # do you want to register to an atlas or inter-subject? Or just ask us, we can guide you ;-)

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageRegistration:\n' \
            .format(self=self)
