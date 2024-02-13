import logging
import time
import unittest

import numpy as np

from neurolib.models.wc import WCModel
from neurolib.models.jax.wc import WCModel as WCModel_jax
from neurolib.utils.loadData import Dataset


class TestWC_jax(unittest.TestCase):
    """
    Basic test for WC model in JAX.
    """

    def test_single_node_deterministic(self):
        logging.info("\t > WC jax: Testing single node ...")
        start = time.time()
        model = WCModel(seed=0)
        model.params["duration"] = 1.0 * 1000
        model.params["sigma_ou"] = 0.0

        model.run()

        model_jax = WCModel_jax(seed=0)
        model_jax.params["duration"] = 1.0 * 1000
        model_jax.params["sigma_ou"] = 0.0

        model_jax.run()

        self.assertTrue(np.allclose(model_jax.exc, model.exc))
        self.assertTrue(np.allclose(model_jax.inh, model.inh))

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))

    def test_single_node_dist(self):
        logging.info("\t > WC jax: Testing activity dist of single node ...")
        start = time.time()

        model = WCModel()
        model.params["duration"] = 5.0 * 1000
        model.params["sigma_ou"] = 0.01

        model.run()

        model_jax = WCModel_jax()
        model_jax.params["duration"] = 5.0 * 1000
        model_jax.params["sigma_ou"] = 0.01

        model_jax.run()

        model_jax_different = WCModel_jax()
        model_jax_different.params["duration"] = 5.0 * 1000
        model_jax_different.params["sigma_ou"] = 0.015

        model_jax_different.run()

        bins = np.logspace(np.log10(0.001), np.log10(1.0), 50)

        model_hist, _ = np.histogram(model.exc.flatten(), bins=bins)
        model_jax_hist, _ = np.histogram(model_jax.exc.flatten(), bins=bins)
        model_jax_different_hist, _ = np.histogram(model_jax_different.exc.flatten(), bins=bins)

        self.assertTrue((np.abs(model_hist - model_jax_hist).sum() / model.exc.shape[1]) < 0.25)
        self.assertTrue((np.abs(model_hist - model_jax_different_hist).sum() / model.exc.shape[1]) > 0.25)

        model_hist_inh, _ = np.histogram(model.inh.flatten(), bins=bins)
        model_jax_hist_inh, _ = np.histogram(model_jax.inh.flatten(), bins=bins)
        model_jax_different_hist_inh, _ = np.histogram(model_jax_different.inh.flatten(), bins=bins)

        self.assertTrue((np.abs(model_hist_inh - model_jax_hist_inh).sum() / model.exc.shape[1]) < 0.25)
        self.assertTrue((np.abs(model_hist_inh - model_jax_different_hist_inh).sum() / model.exc.shape[1]) > 0.25)

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))

    def test_network(self):
        logging.info("\t > WC jax: Testing brain network (chunkwise integration and BOLD simulation) ...")
        start = time.time()
        ds = Dataset("gw")
        model = WCModel(Cmat=ds.Cmat, Dmat=ds.Dmat, seed=0)

        model.params["signalV"] = 4.0
        model.params["duration"] = 10 * 1000
        model.params["sigma_ou"] = 0.0
        model.params["K_gl"] = 0.6

        # local node input parameter
        model.params["exc_ext"] = 0.72

        model.run(chunkwise=True, bold=True, append_outputs=True, chunksize=20000)

        model_jax = WCModel_jax(Cmat=ds.Cmat, Dmat=ds.Dmat, seed=0)

        model_jax.params["signalV"] = 4.0
        model_jax.params["duration"] = 10 * 1000
        model_jax.params["sigma_ou"] = 0.0
        model_jax.params["K_gl"] = 0.6

        # local node input parameter
        model_jax.params["exc_ext"] = 0.72

        model_jax.run(chunkwise=True, bold=True, append_outputs=True, chunksize=20000)

        # jit changes the exact numerics of outputs
        self.assertTrue(np.allclose(model.BOLD.BOLD, model_jax.BOLD.BOLD, rtol=1e-3))

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))
