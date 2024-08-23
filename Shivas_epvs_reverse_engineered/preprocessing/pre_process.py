import SimpleITK as sitk
import numpy as np

def resize_image(image, target_shape):
    """Resizes the 3D image to the target shape using SimpleITK."""
    original_size = np.array(image.GetSize())
    target_size = np.array(target_shape)
    resize_factor = target_size / original_size
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(sz) for sz in target_size])
    resampler.SetOutputSpacing(image.GetSpacing() / resize_factor)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    return resampler.Execute(image)

def normalize_image(image, brain_mask=None):
    """Normalizes voxel values to [0, 1] using min-max normalization."""
    image_array = sitk.GetArrayFromImage(image)
    if brain_mask is not None:
        mask_array = sitk.GetArrayFromImage(brain_mask)
        brain_voxels = image_array[mask_array > 0]
    else:
        brain_voxels = image_array

    # Calculate the 99th percentile value for normalization
    max_val = np.percentile(brain_voxels, 99)
    min_val = np.min(brain_voxels)

    # Clip values to avoid hot spots and scale to [0, 1]
    normalized_array = np.clip((image_array - min_val) / (max_val - min_val), 0, 1)

    return sitk.GetImageFromArray(normalized_array)

def center_brain(image, brain_mask=None):
    """Centers the brain within the image using the brain mask if available."""
    image_array = sitk.GetArrayFromImage(image)
    if brain_mask is not None:
        mask_array = sitk.GetArrayFromImage(brain_mask)
        brain_indices = np.array(np.nonzero(mask_array))
    else:
        brain_indices = np.array(np.nonzero(image_array))

    brain_center = np.mean(brain_indices, axis=1).astype(int)
    image_center = np.array(image_array.shape) // 2

    # Compute shifts needed to center the brain
    shifts = image_center - brain_center

    # Apply shifts using numpy roll
    centered_array = np.roll(image_array, shifts, axis=(0, 1, 2))

    return sitk.GetImageFromArray(centered_array)

def process_nifti_image(input_path, output_path, target_shape=(160, 214, 176), apply_brain_mask=True):
    # Load the NIfTI image using SimpleITK
    nifti_img = sitk.ReadImage(input_path)

    # (Optional) Load brain mask
    if apply_brain_mask:
        brain_mask_path = input_path.replace(".nii", "_mask.nii")
        try:
            brain_mask = sitk.ReadImage(brain_mask_path)
        except:
            brain_mask = None
    else:
        brain_mask = None

    # Resize the image to target shape
    resized_image = resize_image(nifti_img, target_shape)

    # Normalize voxel values
    normalized_image = normalize_image(resized_image, brain_mask)

    # Center the brain within the image
    centered_image = center_brain(normalized_image, brain_mask)

    # Set metadata (direction, spacing, origin) from the original image
    centered_image.SetDirection(nifti_img.GetDirection())
    centered_image.SetSpacing([1.0, 1.0, 1.0])  # Set isotropic spacing (1mm³)
    centered_image.SetOrigin(nifti_img.GetOrigin())

    # Save the processed image
    sitk.WriteImage(centered_image, output_path)

if __name__ == "__main__":
    input_path = "input_image.nii"
    output_path = "output_image.nii"
    process_nifti_image(input_path, output_path)
