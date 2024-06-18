#!/bin/env python3

import glob
import json
import os
import tarfile
from pathlib import Path

import numpy as np
from scipy.io import mmread

_conv_symm = {
    "general": "unsymmetric",
    "symmetric": "symmetric",
}


def mmreadHeader(firstline: str) -> dict[str, str] | None:
    """
    Finds the matrix type and symmetric characteristics from the given line (first of the file in the mtx format)
    """
    line = firstline.removeprefix("%%MatrixMarket ")
    split = line.split(" ")
    if split[1].strip(" \n") != "coordinate":
        return None
    element_type = split[2].strip(" \n")
    symmetry = _conv_symm[split[3].strip(" \n")]

    return {"type": element_type, "symmetry": symmetry}


def mmreadContent(line: str) -> dict[str, float]:
    """
    Finds N and a temporary NNZ from the given line (first after the header in the mtx format)
    """
    split = line.split(" ")
    n = int(split[0].strip())
    nnz = int(split[2].strip())

    return {"n": n, "nnz": nnz, "density": nnz / (n * n)}


def computeBW(file: Path) -> float:
    """
    Computes the matrix maximum bandwith
    """
    mat = mmread(file).tocsr()
    max = np.NINF

    x1 = mat.indptr[0]
    for x in mat.indptr[1:]:
        bw = (x - 1) - x1
        max = bw if bw > max else max
        x1 = x

    return max


def mmreadInfo(file: Path) -> dict | None:
    """
    Parses useful information from a mtx file and stores it in a dictionnary
    """
    res = dict()
    res["name"] = file.name.split(".", maxsplit=1)[0]
    res["path"] = str(file.absolute())
    with open(file, "r") as fd:
        header = mmreadHeader(fd.readline())
        if header is None:
            return None
        line = fd.readline()
        while line[0] == "%":
            line = fd.readline()
        else:
            content = mmreadContent(line)

    # Update nnz for symmetric case
    if header["symmetry"] == "symmetric":
        content["nnz"] = 2 * content["nnz"] - content["n"]

    # Get the max bandwidth
    bw = computeBW(file) / content["n"]
    res["matrix_properties"] = header | content | {"bw": bw}

    return res


def extractAll():
    """
    Extract all .tar.gz file in the current directory
    """
    archives = glob.glob("*.tar.gz")
    for archive in archives:
        with tarfile.open(archive, "r:gz") as file:
            file.extractall()


def main():
    extractAll()
    for entry in os.scandir():
        if entry.is_dir():
            dir_path = Path(entry.path)
            for mtx in dir_path.glob("*.mtx"):
                matrix_info = mmreadInfo(mtx)

                if matrix_info is not None:
                    with open(dir_path / "info.json", "w") as fd:
                        json.dump(matrix_info, fd, indent="\t")


if __name__ == "__main__":
    main()
