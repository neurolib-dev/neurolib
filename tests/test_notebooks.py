import os
import unittest
from shutil import rmtree

from treon.test_execution import execute_notebook


def _run(notebook):
    # from https://github.com/ReviewNB/treon/blob/master/tests/test_execution.py
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), notebook)
    return execute_notebook(path)


class TestExampleNotebooks(unittest.TestCase):
    def test_example_0(self):
        fname = "../examples/example-0-aln-minimal.ipynb"
        successful, output = _run(fname)
        assert successful, print(output)

    def test_example_0_1(self):
        fname = "../examples/example-0.1-hopf-minimal.ipynb"
        successful, output = _run(fname)
        assert successful, print(output)

    def test_example_0_2(self):
        fname = "../examples/example-0.2-basic_analysis.ipynb"
        successful, output = _run(fname)
        assert successful, print(output)

    def test_example_0_3(self):
        fname = "../examples/example-0.3-fhn-minimal.ipynb"
        successful, output = _run(fname)
        assert successful, print(output)

    def test_example_0_4(self):
        fname = "../examples/example-0.4-wc-minimal.ipynb"
        successful, output = _run(fname)
        assert successful, print(output)

    def test_example_1(self):
        fname = "../examples/example-1-aln-parameter-exploration.ipynb"
        successful, output = _run(fname)
        assert successful, print(output)

    def test_example_1_1(self):
        fname = "../examples/example-1.1-custom-parameter-exploration.ipynb"
        successful, output = _run(fname)
        assert successful, print(output)

    def test_example_1_2(self):
        fname = "../examples/example-1.2-brain-network-exploration.ipynb"
        successful, output = _run(fname)
        assert successful, print(output)

    def test_example_1_2_1(self):
        fname = "../examples/example-1.2.1-brain-exploration-postprocessing.ipynb"
        successful, output = _run(fname)
        assert successful, print(output)

    def test_example_2(self):
        fname = "../examples/example-2-evolutionary-optimization-minimal.ipynb"
        successful, output = _run(fname)
        assert successful, print(output)

    def test_example_2_1(self):
        fname = "../examples/example-2.1-evolutionary-optimization-aln.ipynb"
        successful, output = _run(fname)
        assert successful, print(output)

    def test_example_2_2(self):
        fname = "../examples/example-2.2-evolution-brain-network-aln-resting-state-fit.ipynb"
        successful, output = _run(fname)
        assert successful, print(output)

    @classmethod
    def tearDownClass(cls):
        """
        Clear after tests
        """
        rmtree("data")


if __name__ == "__main__":
    unittest.main()
