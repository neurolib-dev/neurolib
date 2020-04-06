import matplotlib.pyplot as plt
import numpy as np

import neurolib.utils.functions as func


def plot_outputs(model, bold_transient=10000):
    _, axs = plt.subplots(2, 3, figsize=(12, 6))

    if "t" in model.outputs:
        axs[0, 0].plot(model.outputs.t, model.output.T)
    if "BOLD" in model.outputs:
        axs[1, 0].plot(
            model.outputs.BOLD.t_BOLD[model.outputs.BOLD.t_BOLD > bold_transient],
            model.outputs.BOLD.BOLD[:, model.outputs.BOLD.t_BOLD > bold_transient].T,
        )
        axs[1, 1].imshow(func.fc(model.outputs.BOLD.BOLD[:, model.outputs.BOLD.t_BOLD > bold_transient]))
    plt.show()


def model_fit(model, ds, bold_transient=10000, fc=True, fcd=False):
    result = {}
    if fc:
        result["fc_scores"] = [
            func.matrix_correlation(func.fc(model.BOLD.BOLD[:, model.BOLD.t_BOLD > bold_transient]), fc)
            for i, fc in enumerate(ds.FCs)
        ]
        result["mean_fc_score"] = np.mean(result["fc_scores"])

    if fcd:
        fcd_scores = [
            func.ts_kolmogorov(model.BOLD.BOLD[:, model.BOLD.t_BOLD > bold_transient], ds.BOLDs[i])
            for i in range(len(ds.BOLDs))
        ]
        fcd_meanScore = np.mean(fcd_scores)

        result["fcd"] = fcd_scores
        result["mean_fcd"] = fcd_meanScore

    return result
