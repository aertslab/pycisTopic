import logging
from typing import Iterator, Literal

import numpy as np
import numpy.typing as npt


def _alocate_chunk_array(
    n_cells: int,
    n_regions: int,
    chunk_size: int,
    chunk_along: Literal["cell", "region"],
    log: logging.Logger
) -> npt.NDArray[np.float32]:
    if chunk_along == "region":
        log.info(
            f"Allocate {(chunk_size * n_cells * 4 / 1024**3):.3f} GiB of RAM for "
            f"calculating (partial) imputed accessibility per cell for ({n_cells}) cells "
            f"for chunk of {chunk_size} regions."
        )
        return np.empty(
            (chunk_size, n_cells),
            dtype=np.float32
        )
    else:
        log.info(
            f"Allocate {(chunk_size * n_regions * 4 / 1024**3):.3f} GiB of RAM for "
            f"calculating (partial) imputed accessibility per region for ({n_regions}) regions "
            f"for chunk of {chunk_size} cells."
        )
        return np.empty(
            (n_regions, chunk_size),
            dtype=np.float32
        )

def impute_accessibility_chunked(
    region_topic: npt.NDArray[np.float32],
    cell_topic: npt.NDArray[np.float32],
    chunk_size: int,
    chunk_along: Literal["cell", "region"],
    log: logging.Logger
) -> Iterator[npt.NDArray[np.float32]]:
    """
    Impute accessibility in chunks.

    Parameters
    ----------
    region_topic
        Region topic matrix (regions x topics).
    cell_topic
        Cell topic matrix (topic x cells).
    chunk_size
        The size of the chunks.
    chunk_along
        Whether to chunk along cells or regions.
    log
        logging.Logger

    Yields
    ------
    Numpy array with imputed accessibility of chunk.

    """
    if region_topic.shape[1] != cell_topic.shape[0]:
        raise ValueError(
            "region- and cell-topic dimensions do not match"
            f" region_topic: {region_topic.shape[1]}"
            f" cell_topic: {cell_topic.shape[0]}."
            "Maybe you have to transpose region_topic or cell_topic"
        )

    if chunk_along not in ["cell", "region"]:
        raise ValueError(f"chunk along should be either 'cell' or 'region', not {chunk_along}!")

    log.info(f"Calculating imputed accessibility in chunks across {chunk_along} ...")

    n_regions = region_topic.shape[0]
    n_cells = cell_topic.shape[1]

    imputed_acc_chunk: npt.NDArray[np.float32] = _alocate_chunk_array(
        n_cells=n_cells,
        n_regions=n_regions,
        chunk_size=chunk_size,
        chunk_along=chunk_along,
        log=log
    )

    n_total: int = n_regions if chunk_along == "region" else n_cells

    for chunk_start in range(0, n_total, chunk_size):
        chunk_end = chunk_start + chunk_size

        log.info(
            "Calculate partial imputed accessibility "
            f"{chunk_start}-{chunk_end} (out of {n_total})."
        )

        # get current chunk of regions or cells
        if chunk_along == "region":
            _cell_topic = cell_topic
            _region_topic = region_topic[
                chunk_start: chunk_end
            ]
            current_chunk_size = _region_topic.shape[0]
        else:
            _cell_topic = cell_topic[:, chunk_start: chunk_end]
            _region_topic = region_topic
            current_chunk_size = _cell_topic.shape[1]

        if current_chunk_size < chunk_size:
            del imputed_acc_chunk
            imputed_acc_chunk = _alocate_chunk_array(
                n_cells=n_cells,
                n_regions=n_regions,
                chunk_size=current_chunk_size,
                chunk_along=chunk_along,
                log=log
            )

        log.info(
            f" - Calculate imputed accessibility for the current chunk of {chunk_along}."
        )
        np.matmul(_region_topic, _cell_topic, out=imputed_acc_chunk)
        yield imputed_acc_chunk

