#-*- python -*-
import logging
import numpy as np
import gwcs
from astropy.stats import sigma_clipped_stats

from jwst import datamodels
from jwst.assign_wcs import nirspec
from stdatamodels.jwst.datamodels import dqflags

from jwst.nsclean.lib import NSClean, NSCleanSubarray

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def mask_ifu_slices(input_model, mask):

    """Find pixels located within IFU slices and flag them in the
    mask, so that they do not get used.

    Parameters
    ----------
    input_model : data model object
        science data

    mask : 2D bool array
        input mask that will be updated

    Returns
    -------
    mask : 2D bool array
        output mask with additional flags for science pixels
    """

    log.info("Finding slice pixels for an IFU image")

    # Initialize global DQ map to all zero (OK to use)
    dqmap = np.zeros_like(input_model.dq)

    # Get the wcs objects for all IFU slices
    list_of_wcs = nirspec.nrs_ifu_wcs(input_model)

    # Loop over the IFU slices, finding the valid region for each
    for (k, ifu_wcs) in enumerate(list_of_wcs):

        # Construct array indexes for pixels in this slice
        x, y = gwcs.wcstools.grid_from_bounding_box(ifu_wcs.bounding_box,
                                                    step=(1, 1),
                                                    center=True)
        # Get the world coords for all pixels in this slice;
        # all we actually need are wavelengths
        coords = ifu_wcs(x, y)
        dq = dqmap[y.astype(int), x.astype(int)]
        wl = coords[2]
        # set non-NaN wavelength locations as do not use (one)
        valid = ~np.isnan(wl)
        dq = dq[valid]
        x = x[valid]
        y = y[valid]
        dq[:] = 1

        # Copy DQ for this slice into global DQ map
        dqmap[y.astype(int), x.astype(int)] = dq

    # Now set all non-zero locations in the mask to False (do not use)
    mask[dqmap==1] = False

    return mask


def mask_slits(input_model, mask):

    """Find pixels located within MOS or fixed slit footprints
    and flag them in the mask, so that they do not get used.

    Parameters
    ----------
    input_model : data model object
        science data

    mask : 2D bool array
        input mask that will be updated

    Returns
    -------
    mask : 2D bool array
        output mask with additional flags for slit pixels
    """

    from jwst.extract_2d.nirspec import offset_wcs

    log.info("Finding slit/slitlet pixels")

    # Get the slit-to-msa frame transform from the WCS object
    slit2msa = input_model.meta.wcs.get_transform('slit_frame', 'msa_frame')

    # Loop over the slits, marking all the pixels within each bounding
    # box as False (do not use) in the mask.
    # Note that for 3D masks (TSO mode), all planes will be set to the same value.
    for slit in slit2msa.slits:
        slit_wcs = nirspec.nrs_wcs_set_input(input_model, slit.name)
        xlo, xhi, ylo, yhi = offset_wcs(slit_wcs)
        mask[..., ylo:yhi, xlo:xhi] = False

    return mask


def create_mask(input_model, mask_spectral_regions, n_sigma):
    """Create the pixel mask needed for setting which pixels to use
    for measuring 1/f noise.

    Parameters
    ----------
    input_model : data model object
        science data

    mask_spectral_regions : bool
        mask slit/slice regions defined in WCS

    n_sigma : float
        sigma threshold for masking outliers

    Returns
    -------
    mask : 2D or 3D bool array
        image mask

    nan_pix : array
        indexes of image locations with NaN values
    """
    exptype = input_model.meta.exposure.type.lower()

    # Initialize mask to all True. Subsequent operations will mask
    # out pixels that contain signal.
    # Note: mask will be 3D for BOTS mode data
    mask = np.full(np.shape(input_model.dq), True)

    # If IFU, mask all pixels contained in the IFU slices
    if exptype == 'nrs_ifu' and mask_spectral_regions:
        mask = mask_ifu_slices(input_model, mask)

    # If MOS or FS, mask all pixels affected by open slitlets
    if exptype in ['nrs_fixedslit', 'nrs_brightobj', 'nrs_msaspec'] and mask_spectral_regions:
        mask = mask_slits(input_model, mask)

    # If IFU or MOS, mask pixels affected by failed-open shutters
    if exptype in ['nrs_ifu', 'nrs_msaspec']:
        open_pix = input_model.dq & dqflags.pixel['MSA_FAILED_OPEN']
        mask[open_pix > 0] = False

    # Temporarily reset NaN pixels and mask them.
    # Save the list of NaN pixel coords, so that they can be reset at the end.
    nan_pix = np.isnan(input_model.data)
    input_model.data[nan_pix] = 0
    mask[nan_pix] = False

    # If IFU or MOS, mask the fixed-slit area of the image; uses hardwired indexes
    if exptype == 'nrs_ifu':
        log.info("Masking the fixed slit region for IFU data.")
        mask[922:1116, :] = False
    elif exptype == 'nrs_msaspec':
        # check for any slits defined in the fixed slit quadrant:
        # if there is nothing there of interest, mask the whole FS region
        slit2msa = input_model.meta.wcs.get_transform('slit_frame', 'msa_frame')
        is_fs = [s.quadrant == 5 for s in slit2msa.slits]
        if not any(is_fs):
            log.info("Masking the fixed slit region for MOS data.")
            mask[922:1116, :] = False
        else:
            log.info("Fixed slits found in MSA definition; "
                     "not masking the fixed slit region for MOS data.")

    # Use left/right reference pixel columns (first and last 4). Can only be
    # applied to data that uses all 2048 columns of the detector.
    if mask.shape[-1] == 2048:
        mask[..., :, :5] = True
        mask[..., :, -4:] = True

    # Mask outliers using sigma clipping stats.
    # For BOTS mode, which uses 3D data, loop over each integration separately.
    if len(input_model.data.shape) == 3:
        for i in range(input_model.data.shape[0]):
            _, median, sigma = sigma_clipped_stats(input_model.data[i], mask=~mask[i], mask_value=0, sigma=5.0)
            outliers = input_model.data[i] > (median + n_sigma * sigma)
            mask[i][outliers] = False
    else:
        _, median, sigma = sigma_clipped_stats(input_model.data, mask=~mask, mask_value=0, sigma=5.0)
        outliers = input_model.data > (median + n_sigma * sigma)
        mask[outliers] = False


    # Return the mask and the record of which pixels were NaN in the input;
    # it'll be needed later
    return mask, nan_pix


def clean_full_frame(detector, image, mask):
    """Clean a full-frame (2048x2048) image.

    Parameters
    ----------
    detector : str
        The name of the detector from which the data originate.

    image : 2D float array
        The image to be cleaned.

    mask : 2D bool array
        The mask that indicates which pixels are to be used in fitting.

    Returns
    -------
    cleaned_image : 2D float array
        The cleaned image.
    """

    # Instantiate the cleaner
    cleaner = NSClean(detector, mask)

    # Clean the image
    try:
        cleaned_image = cleaner.clean(image, buff=True)
    except np.linalg.LinAlgError:
        log.warning("Error cleaning image; step will be skipped")
        return None

    return cleaned_image


def clean_subarray(detector, image, weight, npix_iter=512,
                   exclude_outliers=True, sigrej=4, minfrac=0.05,
                   align=True, corr_inplace=True,
                   fc=(1061, 1211, 49943, 49957)):
    """Clean a subarray image.

    Parameters
    ----------
    detector : str
        The name of the detector from which the data originate.

    image : 2D float array
        The image to be cleaned.

    weight : 2D float array
        Weights for pixels to be used in fitting for 1/f noise.  Pixels
        with zero weight will not be used.

    npix_iter : int
        Number of pixels to process simultaneously.  Default 512.  Should
        be at least a few hundred to access sub-kHz frequencies in areas
        where most pixels are available for fitting.  Previous default
        behavior corresponds to npix_iter of infinity.

    exclude_outliers : bool
        Find and mask outliers in the fit?  Default True

    sigrej : float
        Number of sigma to clip when identifying outliers.  Default 4.

    minfrac : float
        Minimum fraction of pixels locally available in the mask in
        order to attempt a correction.  Default 0.05 (i.e., 5%).

    align : bool
        Flip and rotate the image based on the detector argument?
        Default True.
    
    corr_inplace : bool
        Apply the 1/f correction in place?  If not, return the model
        of 1/f noise rather than the corrected image.

    fc : tuple of four floats
        Cut-on and cut-off frequencies for 1/f correction.  Full
        correction will be applied for frequencies <fc[0] and >fc[-1];
        no correction will be applied for frequencies >fc[1] and <fc[2].
        Default (1061, 1211, 49943, 49957).  Units are Hz, pixel rate is
        100 kHZ, or 100000.
    
    Returns
    -------
    cleaned_image : 2D float array
        The cleaned image.
    """

    # Flip the image to detector coords. NRS1 requires a transpose
    # of the axes, while NRS2 requires a transpose and flip.
    if align:
        if detector == "NRS1":
            image = image.transpose()
            weight = weight.transpose()
        elif detector == "NRS2":
            image = image.transpose()[::-1]
            weight = weight.transpose()[::-1]

    weight = weight*1.
    mask = weight != 0
    
    # We must do the masking of discrepant pixels here: it just
    # doesn't work if we wait and do it in the cleaner.  This is
    # basically copied from lib.py.  Use a robust estimator for
    # standard deviation, then exclude discrepant pixels and their
    # four nearest neighbors from the fit.
    
    if exclude_outliers:
        med = np.median(image[mask])
        std = 1.4825*np.median(np.abs((image - med)[mask]))
        outlier = mask&(np.abs(image - med) > sigrej*std)
        
        mask = mask&(~outlier)
        
        # also get four nearest neighbors of flagged pixels
        mask[1:] = mask[1:]&(~outlier[:-1])
        mask[:-1] = mask[:-1]&(~outlier[1:])
        mask[:, 1:] = mask[:, 1:]&(~outlier[:, :-1])
        mask[:, :-1] = mask[:, :-1]&(~outlier[:, 1:])
            
    weight[~mask] = 0
        
    # Used to determine the fitting intervals along the slow scan
    # direction.  Pre-pend a zero so that we sum_mask[i] is equal
    # to np.sum(mask[:i], axis=1).

    sum_mask = np.array([0] + list(np.cumsum(np.sum(mask, axis=1))))
    
    i1 = 0
    i1_vals = []
    di_list = []
    models = []
    while i1 < image.shape[0] - 1:

        # Want npix_iter available pixels in this section.  If
        # there are fewer than 1.5*npix_iter available pixels in
        # the rest of the image, just go to the end.
        
        for k in range(i1 + 1, image.shape[0] + 1):
            if sum_mask[k] - sum_mask[i1] > npix_iter and sum_mask[-1] - sum_mask[i1] > 1.5*npix_iter:
                break
            
        di = k - i1

        i1_vals += [i1]
        di_list += [di]

        # Fit this section only if at least minpct% of the pixels
        # are available for finding the background.  Don't flag
        # outliers section-by-section; we have to do that earlier
        # over the full array to get reliable values for the mean
        # and standard deviation.
        
        if np.mean(mask[i1:i1 + di]) > minfrac:
            cleaner = NSCleanSubarray(image[i1:i1 + di], weight[i1:i1 + di],
                                      exclude_outliers=False, fc=fc)
            models += [cleaner.clean(return_model=True)]
        else:
            log.warning("Insufficient reference pixels for NSClean around "
                        "row %d; no correction will be made here." % (i1))
            models += [np.zeros(image[i1:i1 + di].shape)]

        # If we have reached the end of the array, we are finished.
        if k == image.shape[0]:
            break

        # Step forward by half an interval so that we have
        # overlapping fitting regions.
        
        i1 += max(int(np.round(di/2)), 1)
            
    model = image*0
    tot_wgt = image*0

    # When we combine different corrections computed over
    # different intervals, each one the highest weight towards the
    # center of its interval and less weight towards the edge.
    # Use nonzero weights everywhere so that if only one
    # correction is available it gets unit weight when we
    # normalize.
    
    for i in range(len(models)):
        wgt = 1.001 - np.abs(np.linspace(-1, 1, di_list[i]))[:, np.newaxis]
        model[i1_vals[i]:i1_vals[i] + di_list[i]] += wgt*models[i]
        tot_wgt[i1_vals[i]:i1_vals[i] + di_list[i]] += wgt
        
    model /= tot_wgt

    if corr_inplace:
        cleaned_image = image - model
        
        # Restore the cleaned image to the science frame
        if align:
            if detector == "NRS1":
                cleaned_image = cleaned_image.transpose()
            elif detector == "NRS2":
                cleaned_image = cleaned_image[::-1].transpose()
            
        return cleaned_image

    else:
        if align:
            if detector == "NRS1":
                model = model.transpose()
            elif detector == "NRS2":
                model = model[::-1].transpose()
        return model
        

def do_correction(input_model, mask_spectral_regions, n_sigma, save_mask, user_mask,
                  npix_iter=1024, exclude_outliers=True, sigrej=3,
                  minfrac_subarray=0.1, minfrac_fullarray=0.3,
                  expand_mask=True, filter_sig=5, threshold=0.2,
                  aggressiveness=0):
    """Apply the NSClean 1/f noise correction

    Parameters
    ----------
    input_model : data model object
        science data to be corrected

    mask_spectral_regions : bool
        Mask slit/slice regions defined in WCS

    n_sigma : float
        n-sigma rejection level for finding outliers

    save_mask : bool
        switch to indicate whether the mask should be saved

    user_mask : str or None
        Path to user-supplied mask image
    
    npix_iter : int
        Number of pixels to process simultaneously.  Default 512.  Should
        be at least a few hundred to access sub-kHz frequencies in areas
        where most pixels are available for fitting.  Previous default
        behavior corresponds to npix_iter of infinity.

    exclude_outliers : bool
        Find and mask outliers in the fit?  Default True

    sigrej : float
        Number of sigma to clip when identifying outliers.  Default 4.

    minfrac_subarray : float
        Minimum fraction of pixels locally available in the mask in
        order to attempt a correction for a subarray.  Default 0.1
        (i.e., 10%).

    minfrac_fullarray : float
        Minimum fraction of pixels locally available in the mask in
        order to attempt a correction for a full frame.  Default 0.3
        (i.e., 30%).

    expand_mask : bool
        Locally expand the mask to flag pixels where a large fraction of
        neighbors are flagged as illuminated.  Mitigates the problem of
        using too many pixels near bright objects.  Only used in full
        array mode.  Default True.

    filter_sig : float
        Radius to use for looking for flagged neighboring pixels.  Only
        used if expand_mask is True.  Default 5.
    
    threshold : float
        Maximum fraction of neighbors that have illumination for a pixel
        to remain available for 1/f fitting.  Only used if expand_mask is
        True.  Default 0.2.

    aggressiveness : float
        How aggressive a 1/f subtraction should we do?  Goes from a
        minimum of zero (which will be similar to the median-based
        correction) to 11, which will suppress significantly more noise
        if there is a good background model and much of the frame is
        read-noise-limited.

    Returns
    -------
    output_model : `~jwst.datamodel.JwstDataModel`
        corrected data

    mask_model : `~jwst.datamodel.JwstDataModel`
        pixel mask to be saved or None
    """

    detector = input_model.meta.instrument.detector.upper()
    exp_type = input_model.meta.exposure.type
    log.info(f'Input exposure type is {exp_type}, detector={detector}')

    # Check for a valid input that we can work on
    if input_model.meta.subarray.name.upper() == "ALLSLITS":
        log.warning("Step cannot be applied to ALLSLITS subarray images")
        log.warning("Step will be skipped")
        input_model.meta.cal_step.nsclean = 'SKIPPED'
        return input_model, None

    output_model = input_model.copy()

    if not (aggressiveness >= 0 and aggressiveness <= 11):
        raise ValueError("Argument aggressiveness to do_correction should be "
                         "between 0 and 11 (inclusive)")
    
    # Check for a user-supplied mask image. If so, use it.
    if user_mask is not None:
        mask_model = datamodels.open(user_mask)
        Mask = (mask_model.data.copy()).astype(np.bool_)

        # Reset and save list of NaN pixels in the input image
        nan_pix = np.isnan(input_model.data)
        input_model.data[nan_pix] = 0
        Mask[nan_pix] = False

    else:
        # Create the pixel mask that'll be used to indicate which pixels
        # to include in the 1/f noise measurements. Basically, we're setting
        # all illuminated pixels to False, so that they do not get used, and
        # setting all unilluminated pixels to True (so they DO get used).
        # For BOTS mode the mask will be 3D, to accommodate changes in masked
        # pixels per integration.
        log.info("Creating mask")
        Mask, nan_pix = create_mask(input_model, mask_spectral_regions, n_sigma)

    log.info(f"Cleaning image {input_model.meta.filename}")

    # Setup for handling 2D or 3D inputs
    if len(input_model.data.shape) == 3:
        nints = input_model.data.shape[0]
        # Check for 3D mask
        if len(Mask.shape) == 2:
            log.warning("Data are 3D, but mask is 2D. Step will be skipped.")
            output_model.meta.cal_step.nsclean = 'SKIPPED'
            return output_model, None
    else:
        nints = 1

    # Loop over integrations (even if there's only 1)
    for i in range(nints):
        log.debug(f" working on integration {i+1}")
        if len(input_model.data.shape) == 3:
            image = np.float32(input_model.data[i])
            mask = Mask[i]
        else:
            image = np.float32(input_model.data)
            mask = Mask

        if input_model.data.shape[-2:] == (2048, 2048):
            # Clean a full-frame image.  I will do this in two stages.
            # First, I will use the four output channels combined to
            # fit the low-ish frequencies.  Then I will use each channel
            # to fit the very lowest frequencies of noise.

            f1a = aggressiveness*600/11 + 10
            f1b = aggressiveness*700/11 + 20
            if aggressiveness > 5:
                f2a, f2b = 49943, 49957
            else:
                f2a, f2b = 51000, 52000
            if aggressiveness > 10:
                subchannelmean = True
            else:
                subchannelmean = False
                
            cleaned_image, mask = clean_full_frame_alt(detector, image, mask,
                                                       exclude_outliers=exclude_outliers, sigrej=sigrej, npix_iter=npix_iter, minfrac=minfrac_fullarray,
                                                       expand_mask=expand_mask, filter_sig=filter_sig, threshold=threshold, subchannelmean=subchannelmean, fc=(f1a, f1b, f2a, f2b))

            if aggressiveness >= 7:
                cleaned_image, mask = clean_full_frame_alt(detector, cleaned_image, mask,
                                                           npix_iter=npix_iter,
                                                           minfrac=minfrac_fullarray,
                                                           chan_wgt=20,
                                                           exclude_outliers=False,
                                                           expand_mask=False,
                                                           subchannelmean=subchannelmean,
                                                           fc=(90, 100, 51000, 52000))

        else:
            # Clean a subarray image
            cleaned_image = clean_subarray(detector, image, mask,
                                           minfrac=minfrac_subarray)

        # Check for failure
        if cleaned_image is None:
            output_model.meta.cal_step.nsclean = 'SKIPPED'
            break
        else:
            # Store the cleaned image in the output model
            if len(output_model.data.shape) == 3:
                output_model.data[i] = cleaned_image
            else:
                output_model.data = cleaned_image
                
        if len(input_model.data.shape) == 3:
            Mask[i] = mask
        else:
            Mask = mask

    # Store the mask image in a model, if requested
    if save_mask:
        if len(Mask.shape) == 3:
            mask_model = datamodels.CubeModel(data=Mask)
        else:
            mask_model = datamodels.ImageModel(data=Mask)
    else:
        mask_model = None
                
    # Restore NaN's from original image
    output_model.data[nan_pix] = np.nan

    # Set completion status
    output_model.meta.cal_step.nsclean = 'COMPLETE'

    return output_model, mask_model



def clean_full_frame_alt(detector, image, mask, npix_iter=1024, chan_wgt=1,
                         exclude_outliers=True, sigrej=3, minfrac=0.3, 
                         expand_mask=True, filter_sig=5, threshold=0.2, 
                         subchannelmean=True, fc=(600, 700, 49943, 49957)):

    """Clean a full-frame (2048x2048) image.

    Parameters
    ----------
    detector : str
        The name of the detector from which the data originate.

    image : 2D float array
        The image to be cleaned.

    mask : 2D bool array
        Boolean mask of pixels to be used to compute the 1/f correction.

    npix_iter : int
        Number of pixels to process simultaneously.  Default 512.  Should
        be at least a few hundred to access sub-kHz frequencies in areas
        where most pixels are available for fitting.  Previous default
        behavior corresponds to npix_iter of infinity.

    chan_wgt : float
        How strongly to weight a channel relative to the other channels
        when computing a correction.  chan_wgt=1 means weight all
        channels equally, so that the same 1/f correction applies to each
        of them.  chan_wgt=infinity means only use this channel for its
        correction.  Default 1.
    
    exclude_outliers : bool
        Find and mask outliers in the fit?  Default True

    sigrej : float
        Number of sigma to clip when identifying outliers.  Default 4.

    minfrac : float
        Minimum fraction of pixels locally available in the mask in
        order to attempt a correction.  Default 0.3 (i.e., 30%).

    expand_mask : bool
        Locally expand the mask to flag pixels where a large fraction of
        neighbors are flagged as illuminated.  Mitigates the problem of
        using too many pixels near bright objects.  Only used in full
        array mode.  Default True.

    filter_sig : float
        Radius to use for looking for flagged neighboring pixels.  Only
        used if expand_mask is True.  Default 5.
    
    threshold : float
        Maximum fraction of neighbors that have illumination for a pixel
        to remain available for 1/f fitting.  Only used if expand_mask is
        True.  Default 0.2.

    subchannelmean : bool
        Subtract the mean of each channel over the available unilluminated
        pixels?  Default True. 
    
    fc : tuple of four floats
        Cut-on and cut-off frequencies for 1/f correction.  Full
        correction will be applied for frequencies <fc[0] and >fc[-1];
        no correction will be applied for frequencies >fc[1] and <fc[2].
        Default (600, 700, 49943, 49957).  Units are Hz, pixel rate is
        100 kHz.  
    
    Returns
    -------
    cleaned_image : 2D float array
        The cleaned image.

    mask : 2D boolean array
        The mask of pixels used to compute the correction.

    """
    
    if detector == "NRS1":
        image = image.transpose()
        mask = mask.transpose()
    elif detector == "NRS2":
        image = image.transpose()[::-1]
        mask = mask.transpose()[::-1]

    # Make sure we aren't using the reference pixels; they should have
    # already been used for the SIRS correction and have nothing more
    # to offer.
    
    mask[:4] = mask[-4:] = mask[:, :4] = mask[:, -4:] = False

    # Mask outliers.  This is currently done in Bernie's routine in lib.py,
    # and is moved here, with outlier flagging set to False later.
    
    if exclude_outliers:
        med = np.median(image[mask])
        std = 1.4825*np.median(np.abs((image - med)[mask]))
        outlier = mask&(np.abs(image - med) > sigrej*std)
        
        mask = mask&(~outlier)
        
        # also get four nearest neighbors of flagged pixels
        mask[1:] = mask[1:]&(~outlier[:-1])
        mask[:-1] = mask[:-1]&(~outlier[1:])
        mask[:, 1:] = mask[:, 1:]&(~outlier[:, :-1])
        mask[:, :-1] = mask[:, :-1]&(~outlier[:, 1:])

    # Expand the mask by looking at the neighbors of candidate
    # background pixels.  Pixels survive, and can still be in the
    # mask, if <threshold of their neighbors are flagged as outliers
    # or illuminated.
    
    if expand_mask:
        filter = np.arange(-2*filter_sig, 2*filter_sig + 1)
        filter = np.exp(-filter**2/(2*filter_sig**2))
        filter /= np.sum(filter)
        
        filtered_mask = 1 - mask*1.
        # The array below takes care of edge effects from the incomplete convolution.
        norm = np.ones(mask.shape)
        for i in range(4, mask.shape[0] - 4):
            filtered_mask[i, 4:-4] = np.convolve(filtered_mask[i, 4:-4], filter, mode='same')
            norm[i, 4:-4] = np.convolve(norm[i, 4:-4], filter, mode='same')
        for i in range(4, mask.shape[1] - 4):
            filtered_mask[4:-4, i] = np.convolve(filtered_mask[4:-4, i], filter, mode='same')
            norm[4:-4, i] = np.convolve(norm[4:-4, i], filter, mode='same')

        filtered_mask /= norm

        mask = mask&(filtered_mask < threshold)

    masked = image*mask

    # Compute the correction channel-by-channel.  This only needs to
    # be done once if the channels are being equally weighted.
    
    for ichan in range(4):
        
        # If chan_wgt==1, all channels are weighted equally; the model
        # will be the same for all of them.  In that case we only need
        # to compute it once.  Weights for the unmasked pixels are one
        # for channels other than the one to be fitted, and chan_wgt
        # for the channel being fitted.
        
        if ichan == 0 or chan_wgt != 1:
            weights = mask[:, :512]*0.
            refchan = weights*0.
            for i in range(4):
                if i == ichan:
                    w = chan_wgt
                else:
                    w = 1
                    
                # Readout directions are opposite for alternating channels.
                if i%2 == 0:
                    weights += mask[:, 512*i:512*(i + 1)]*w
                    refchan += masked[:, 512*i:512*(i + 1)]*w
                else:
                    weights += mask[:, 512*i:512*(i + 1)][:, ::-1]*w
                    refchan += masked[:, 512*i:512*(i + 1)][:, ::-1]*w

            # Back to units of counts or count rate for the fitting
            # the reference channel that we have partially filled.
            # Add a very small number to prevent division by zero.
            
            refchan /= weights + 1e-100
            model = clean_subarray(detector, refchan[4:-4], weights[4:-4],
                                   npix_iter=npix_iter,
                                   exclude_outliers=False, minfrac=minfrac,
                                   align=False, corr_inplace=False, fc=fc)

        # Readout directions are opposite for alternating channels.
        # Don't modify the reference pixels on the ends of each
        # channel.
        
        if ichan%2 == 0:
            image[4:-4, ichan*512:(ichan + 1)*512] -= model
        else:
            image[4:-4, ichan*512:(ichan + 1)*512] -= model[:, ::-1]

        if subchannelmean:
            image[4:-4, ichan*512:(ichan + 1)*512] -= np.mean(image[4:-4, ichan*512:(ichan + 1)*512][mask[4:-4, ichan*512:(ichan + 1)*512]])
        
    if detector == "NRS1":
        image = image.transpose()
        mask = mask.transpose()
    elif detector == "NRS2":
        image = image[::-1].transpose()
        mask = mask[::-1].transpose()
            
    return image, mask
