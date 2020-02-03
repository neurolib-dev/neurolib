import os
import unittest

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(notebook_path):
    # From https://www.blog.pythonlibrary.org/2018/10/16/testing-jupyter-notebooks/
    nb_name, _ = os.path.splitext(os.path.basename(notebook_path))
    dirname = os.path.dirname(notebook_path)

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=600, kernel_name="python3")
    proc.allow_errors = True

    proc.preprocess(nb, {})
    output_path = os.path.join(dirname, "{}_all_output.ipynb".format(nb_name))

    with open(output_path, mode="wt") as f:
        nbformat.write(nb, f)
    errors = []
    for cell in nb.cells:
        if "outputs" in cell:
            for output in cell["outputs"]:
                if output.output_type == "error":
                    errors.append(output)
    os.remove(output_path)
    return nb, errors


def print_errors(fname, errors):
    if len(errors) > 0:
        print(f"Error in {fname}")
    for error in errors:
        if "evalue" in error:
            print(error["ename"], error["evalue"])


class TestVanillaEvolution(unittest.TestCase):
    def test_example_0(self):
        fname = "examples/example-0-aln-minimal.ipynb"
        nb, errors = run_notebook(fname)
        print_errors(fname, errors)
        assert len(errors) == 0, f"Error in {fname}"

    def test_example_0_1(self):
        fname = "examples/example-0.1-hopf-minimal.ipynb"
        nb, errors = run_notebook(fname)
        print_errors(fname, errors)
        assert len(errors) == 0, f"Error in {fname}"

    def test_example_1(self):
        fname = "examples/example-1-aln-parameter-exploration.ipynb"
        nb, errors = run_notebook(fname)
        print_errors(fname, errors)
        assert len(errors) == 0, f"Error in {fname}"

    def test_example_2(self):
        fname = "examples/example-2-evolutionary-optimization-minimal.ipynb"
        nb, errors = run_notebook(fname)
        print_errors(fname, errors)
        assert len(errors) == 0, f"Error in {fname}"

    def test_example_3(self):
        fname = "examples/example-3-evolutionary-optimization-aln.ipynb"
        nb, errors = run_notebook(fname)
        print_errors(fname, errors)
        assert len(errors) == 0, f"Error in {fname}"
