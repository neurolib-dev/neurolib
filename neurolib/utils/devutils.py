import matplotlib.pyplot as plt
import numpy as np
import scipy

import neurolib.utils.functions as func


def plot_outputs(
    model,
    ds=None,
    activity_xlim=None,
    bold_transient=10000,
    spectrum_windowsize=1,
    plot_fcd=None,
):

    # check if BOLD signal is long enough for FCD
    FCD_THRESHOLD = 60  # seconds
    # plot_fcd = False
    if "BOLD" in model.outputs and plot_fcd is None:
        if (
            len(model.BOLD.BOLD.T) > FCD_THRESHOLD / 2
        ):  # div by 2 because of bold sampling rate
            plot_fcd = True

    # find out how many rows we will need to plot
    nrows = 1
    if "BOLD" in model.outputs:
        nrows += 1
    if plot_fcd:
        nrows += 1

    fig, axs = plt.subplots(nrows, 3, figsize=(12, nrows * 3), dpi=150)

    if "t" in model.outputs:
        # if there are more than one row
        if nrows > 1:
            ax = axs[0]
        # if only this row is being plotted
        else:
            ax = axs
        ax[0].set_ylabel("Activity")
        ax[0].set_xlabel("Time [s]")
        ax[0].plot(model.outputs.t / 1000, model.output.T, alpha=0.8, lw=0.5)
        ax[0].plot(
            model.outputs.t / 1000,
            np.mean(model.output, axis=0),
            c="r",
            alpha=0.8,
            lw=1.5,
            label="average",
            zorder=3,
        )

        ax[0].plot(
            model.outputs.t / 1000,
            np.mean(model.output[1::2, :], axis=0),
            c="k",
            alpha=0.8,
            lw=1,
            label="L average",
        )

        ax[0].plot(
            model.outputs.t / 1000,
            np.mean(model.output[::2, :], axis=0),
            c="k",
            alpha=0.8,
            lw=1,
            label="R average",
        )

        ax[0].set_xlim(activity_xlim)

        ax[1].set_ylabel("Node")
        ax[1].set_xlabel("Time [s]")
        # plt.imshow(rates_exc*1000, aspect='auto', extent=[0, params['duration'], N, 0], clim=(0, 10))
        ax[1].imshow(
            model.output,
            aspect="auto",
            extent=[0, model.t[-1] / 1000, model.params.N, 0],
            clim=(0, 10),
        )

        # output frequency spectrum
        ax[2].set_ylabel("Power")
        ax[2].set_xlabel("Frequency [Hz]")
        for o in model.output:
            frs, pwrs = getPowerSpectrum(
                o, dt=model.params.dt, spectrum_windowsize=spectrum_windowsize
            )
            ax[2].plot(frs, pwrs, alpha=0.6, lw=0.5)
        frs, pwrs = getMeanPowerSpectrum(
            model.output, dt=model.params.dt, spectrum_windowsize=spectrum_windowsize
        )
        ax[2].plot(frs, pwrs, lw=3, c="springgreen")

        ## frequency spectrum annotations
        # peaks = scipy.signal.find_peaks_cwt(pwrs, np.arange(1, 2))
        peaks, properties = scipy.signal.find_peaks(np.log(pwrs), prominence=0.1)
        for p in peaks:
            ax[2].scatter(frs[p], pwrs[p], c="springgreen", zorder=20)
            # p = np.argmax(Pxxs)
            ax[2].annotate(
                s=f"  {frs[p]:.1f} Hz",
                xy=(frs[p], pwrs[p]),
                fontsize=10,
                color="springgreen",
                alpha=0.8,
                fontweight="bold",
            )

    if "BOLD" in model.outputs:
        fit = model_fit(model, ds, bold_transient, fc=True, fcd=True)
        # BOLD plotting ----------
        axs[1, 0].set_ylabel("BOLD")
        axs[1, 0].set_xlabel("Time [s]")
        t_bold = (
            model.outputs.BOLD.t_BOLD[model.outputs.BOLD.t_BOLD > bold_transient] / 1000
        )
        bold = model.outputs.BOLD.BOLD[:, model.outputs.BOLD.t_BOLD > bold_transient]
        axs[1, 0].plot(t_bold, bold.T, lw=1.5, alpha=0.8)

        axs[1, 1].set_title("FC", fontsize=12)
        if ds is not None:
            fc_fit = fit["mean_fc"]
            axs[1, 1].set_title(f"FC (corr: {fc_fit:0.2f})", fontsize=12)
        axs[1, 1].imshow(func.fc(bold), origin="upper")
        axs[1, 1].set_ylabel("Node")
        axs[1, 1].set_xlabel("Node")

        axs[1, 2].set_title("FC corr over time", fontsize=12)
        axs[1, 2].plot(
            np.arange(4, bold.shape[1] * 2, step=2),
            np.array(
                [
                    [
                        func.matrix_correlation(func.fc(bold[:, :t]), fc)
                        for t in range(2, bold.shape[1])
                    ]
                    for fc in ds.FCs
                ]
            ).T,
        )
        axs[1, 2].set_ylabel("FC fit")
        axs[1, 2].set_xlabel("Simulation time [s]")

        # FCD plotting ------------
        if plot_fcd and "mean_fcd" in fit:
            # plot image of fcd
            axs[2, 0].set_title("FCD", fontsize=12)
            axs[2, 0].set_ylabel("$n_{window}$")
            axs[2, 0].set_xlabel("$n_{window}$")
            axs[2, 0].imshow(func.fcd(bold), origin="upper")

            # plot distribution in fcd
            fcd_fit = fit["mean_fcd"]
            axs[2, 1].set_title(f"FCD distance {fcd_fit:0.2f}", fontsize=12)
            axs[2, 1].set_ylabel("P")
            axs[2, 1].set_xlabel("triu(FCD)")
            m1 = func.fcd(bold)
            triu_m1_vals = m1[np.triu_indices(m1.shape[0], k=1)]
            axs[2, 1].hist(
                triu_m1_vals, density=True, color="springgreen", zorder=10, alpha=0.6
            )
            # plot fcd distributions of data
            if hasattr(ds, "FCDs"):
                for emp_fcd in ds.FCDs:
                    m1 = emp_fcd
                    triu_m1_vals = m1[np.triu_indices(m1.shape[0], k=1)]
                    axs[2, 1].hist(triu_m1_vals, density=True, alpha=0.5)

            # temp bullshit
            axs[2, 2].plot(
                model.outputs.rates_exc[0, :], model.outputs.rates_inh[0, :], lw=0.5
            )
            axs[2, 2].set_xlabel("$r_{exc}$")
            axs[2, 2].set_ylabel("$r_{inh}$")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def model_fit(model, ds, bold_transient=10000, fc=True, fcd=True):
    result = {}
    if fc:
        result["fc"] = [
            func.matrix_correlation(
                func.fc(model.BOLD.BOLD[:, model.BOLD.t_BOLD > bold_transient]), fc
            )
            for i, fc in enumerate(ds.FCs)
        ]
        result["mean_fc"] = np.mean(result["fc"])

    if fcd and model.BOLD.t_BOLD[-1] > 5 * 60 * 1000:
        fcd_sim = func.fcd(model.BOLD.BOLD[:, model.BOLD.t_BOLD > bold_transient])
        # if the FCD dataset is already computed, use it
        if hasattr(ds, "FCDs"):
            fcd_scores = [
                func.matrix_kolmogorov(fcd_sim, fcd_emp,) for fcd_emp in ds.FCDs
            ]
        else:
            fcd_scores = [
                func.ts_kolmogorov(
                    model.BOLD.BOLD[:, model.BOLD.t_BOLD > bold_transient], bold
                )
                for bold in ds.BOLDs
            ]
        fcd_meanScore = np.mean(fcd_scores)

        result["fcd"] = fcd_scores
        result["mean_fcd"] = fcd_meanScore

    return result


def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """Create a view of `array` which for every point gives the n-dimensional
    neighbourhood of size window. New dimensions are added at the end of
    `array` or after the corresponding original dimension.
    
    From: https://gist.github.com/seberg/3866040
    
    Parameters
    ----------
    array : array_like
        Array to which the rolling window is applied.
    window : int or tuple
        Either a single integer to create a window of only the last axis or a
        tuple to create it for the last len(window) axes. 0 can be used as a
        to ignore a dimension in the window.
    asteps : tuple
        Aligned at the last axis, new steps for the original array, ie. for
        creation of non-overlapping windows. (Equivalent to slicing result)
    wsteps : int or tuple (same size as window)
        steps for the added window dimensions. These can be 0 to repeat values
        along the axis.
    axes: int or tuple
        If given, must have the same size as window. In this case window is
        interpreted as the size in the dimension given by axes. IE. a window
        of (2, 1) is equivalent to window=2 and axis=-2.       
    toend : bool
        If False, the new dimensions are right after the corresponding original
        dimension, instead of at the end of the array. Adding the new axes at the
        end makes it easier to get the neighborhood, however toend=False will give
        a more intuitive result if you view the whole array.
    
    Returns
    -------
    A view on `array` which is smaller to fit the windows and has windows added
    dimensions (0s not counting), ie. every point of `array` is an array of size
    window.
    
    Examples
    --------
    >>> a = np.arange(9).reshape(3,3)
    >>> rolling_window(a, (2,2))
    array([[[[0, 1],
             [3, 4]],
            [[1, 2],
             [4, 5]]],
           [[[3, 4],
             [6, 7]],
            [[4, 5],
             [7, 8]]]])
    
    Or to create non-overlapping windows, but only along the first dimension:
    >>> rolling_window(a, (2,0), asteps=(2,1))
    array([[[0, 3],
            [1, 4],
            [2, 5]]])
    
    Note that the 0 is discared, so that the output dimension is 3:
    >>> rolling_window(a, (2,0), asteps=(2,1)).shape
    (1, 3, 2)
    
    This is useful for example to calculate the maximum in all (overlapping)
    2x2 submatrixes:
    >>> rolling_window(a, (2,2)).max((2,3))
    array([[4, 5],
           [7, 8]])
           
    Or delay embedding (3D embedding with delay 2):
    >>> x = np.arange(10)
    >>> rolling_window(x, 3, wsteps=2)
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6],
           [3, 5, 7],
           [4, 6, 8],
           [5, 7, 9]])
    """
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)  # maybe crude to cast to int...

    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.")

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps) :] = asteps

        if np.any(asteps < 1):
            raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps

    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
            raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1  # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window) :] < window * wsteps):
        raise ValueError(
            "`window` * `wsteps` larger then `array` in at least one dimension."
        )

    new_shape = orig_shape  # just renaming...

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window) :] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window) :] * wsteps

    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window) :] = window
        _window = _.copy()
        _[-len(window) :] = new_strides
        _new_strides = _

        new_shape = np.zeros(len(shape) * 2, dtype=int)
        new_strides = np.zeros(len(shape) * 2, dtype=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)
