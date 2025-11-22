import logging
from typing import Iterable, NamedTuple

import anndata as ad
import h5py
import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix
from typing import Iterable

logger = logging.getLogger()
import pathlib
import pandas as pd


class _RowInvairantData(NamedTuple):
    var: pd.DataFrame = None
    raw_var: pd.DataFrame = None
    uns: dict = None


class H5adReader:

    def __init__(self, filename: str | pathlib.Path, chunk_size=1000):
        f = h5py.File(filename, "r")
        self._root: h5py.File = f
        self.chunk_size = chunk_size

        self._row_count = self._calculate_row_count()
        self.row_invariant_data = None

    def _calculate_row_count(self) -> int:
        obs = self._root["obs"]
        index_field = obs.attrs["_index"]
        return len(obs[index_field])

    @property
    def row_count(self):
        return self._row_count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._root.close()

    def __len__(self):
        """ "
        :return: the number of chunks in this h5ad file
        """
        return 1 + (self._row_count - 1) // self.chunk_size

    def __getitem__(self, i: int) -> ad.AnnData:
        if i >= len(self):
            raise IndexError
        start = i * self.chunk_size
        end = start + self.chunk_size
        return self._read_chunk(slice(start, end))

    def _read_chunk(self, row_selector: slice | Iterable[int]) -> AnnData:
        if x_group := self._root.get("X"):
            X = self._read_csr_matrix_chunk(x_group, row_selector)
        else:
            X = None

        if obs_group := self._root["obs"]:
            obs = self._read_dataframe(obs_group, row_selector)
        else:
            obs = None

        raw_dict = self._read_raw_section(row_selector)

        obsm = None
        if obsm_group := self._root.get("obsm"):
            obsm = self._read_dict_chunk(obsm_group, row_selector)

        return ad.AnnData(X=X, obs=obs, var=self.var, raw=raw_dict, uns=self.uns, obsm=obsm)

    def select_rows(self, row_selector: slice | Iterable[int]) -> AnnData:
        if isinstance(row_selector, slice):
            return self._read_chunk(row_selector)

        row_indices = sorted(row_selector)
        n = len(row_indices)
        max_chunk_size = 1000
        num_chunks = (n + max_chunk_size - 1) // max_chunk_size  # Compute number of chunks
        chunk_size = (n + num_chunks - 1) // num_chunks  # Compute chunk size to distribute evenly
        chunks = (row_indices[i : i + chunk_size] for i in range(0, n, chunk_size))

        return ad.concat(self._read_chunk(c) for c in chunks)

    def sample(
        self,
        sample_size: int,
        concat: bool = True,
        held_out_embryos: bool = False,
        every_other_timepoint: bool = False,
    ) -> ad.AnnData | Iterable[ad.AnnData]:
        """
        Uniformly samples a random subset of the rows from the h5ad file.
        :param sample_size: the number of rows to sample
        :param concat: if True, return a single AnnData object with the sampled rows, otherwise return an iterator of small AnnData objects
        """

        def genereate_adata_objects(sample_size: int):
            if sample_size < 0:
                raise ValueError("sample_size must be greater than 0")
            if sample_size >= self.row_count:
                yield from self
                return

            indices = np.random.choice(self.row_count, sample_size, replace=False)
            indices.sort()

            chunk_count = len(self)
            from tqdm import tqdm

            for i in tqdm(range(chunk_count), desc="Sampling chunks"):
                split_idx = np.searchsorted(indices, self.chunk_size)
                # if the chunk is before the first index, skip it
                if split_idx > 0:
                    chunk = self[i]
                    yield chunk[indices[:split_idx]].copy()
                    indices = indices[split_idx:]
                    if len(indices) == 0:
                        break
                indices -= self.chunk_size

        generator = genereate_adata_objects(sample_size)
        if concat:
            return ad.concat(adatas=generator)
        else:
            return generator

    def _read_row_invariant_data(self):
        if self.row_invariant_data is None:
            var_group = self._root.get("var")
            raw_var_group = self._root.get("raw/var")
            uns_group = self._root.get("uns")

            var = ad.io.read_elem(var_group) if var_group else None
            raw_var = ad.io.read_elem(raw_var_group) if raw_var_group else None
            uns = ad.io.read_elem(uns_group) if uns_group else None

            self.row_invariant_data = _RowInvairantData(var=var, raw_var=raw_var, uns=uns)
        return self.row_invariant_data

    @property
    def var(self):
        return self._read_row_invariant_data().var

    @property
    def raw_var(self):
        return self._read_row_invariant_data().raw_var

    @property
    def uns(self):
        return self._read_row_invariant_data().uns

    def _read_raw_section(self, row_selector: slice | Iterable[int]) -> dict:
        raw_dict = {}
        if raw_x := self._root.get("raw/X"):
            raw_dict["X"] = self._read_csr_matrix_chunk(raw_x, row_selector)

        # if group := self._root.get('raw/varm'):
        #     raw_dict['varm'] = self._read_dataframe(group, start, end)

        if self.raw_var is not None and len(self.raw_var) > 0:
            raw_dict["var"] = self.raw_var

        return raw_dict

    def _read_group(self, group: h5py.Group, row_selector: slice | Iterable[int] = slice(None, None)):
        encoding_type = group.attrs["encoding-type"]
        if encoding_type == "csr_matrix":
            return self._read_csr_matrix_chunk(group, row_selector)
        elif encoding_type == "dataframe":
            return self._read_dataframe(group, row_selector)
        elif encoding_type == "dict":
            return self._read_dict_chunk(group)
        else:
            raise ValueError(f"group {group} has an unexpected encoding type: {encoding_type}")

    def _read_csr_matrix_chunk(
        self, group: h5py.Group, row_selector: slice | Iterable[int] = slice(None, None)
    ) -> csr_matrix | np.ndarray:
        encoding_type = group.attrs["encoding-type"]
        if encoding_type == "array":
            return group[row_selector, :]

        indptr = group["indptr"]
        data = group["data"]
        indices = group["indices"]
        full_matrix_shape = group.attrs["shape"]

        if isinstance(row_selector, slice):
            start = row_selector.start or 0
            end = row_selector.stop or full_matrix_shape[0]

            # +1 because an indptr block ends must also contain the end index of the last line
            chunk_indptr = indptr[start : end + 1]
            data_start = chunk_indptr[0]
            data_end = chunk_indptr[-1]

            chunk_data = data[data_start:data_end]
            chunk_indices = indices[data_start:data_end]

            # the indptr array should be made relative to the rows of the chunk, which always start from 0
            chunk_indptr = chunk_indptr - chunk_indptr[0]
        else:
            # row_indices = np.asarray(row_selector, dtype=np.int32)
            # chunk_indptr = np.zeros(len(row_indices) + 1, dtype=indptr.dtype)
            # chunk_data = []
            # chunk_indices = []
            #
            # for i, row in enumerate(row_indices):
            #     start, end = indptr[row], indptr[row + 1]
            #     chunk_data.extend(data[start:end])
            #     chunk_indices.extend(indices[start:end])
            #     chunk_indptr[i + 1] = chunk_indptr[i] + (end - start)
            row_indices = np.asarray(row_selector, dtype=np.int32)
            row_starts = indptr[row_indices]  # Start of each row in the data array
            row_ends = indptr[row_indices + 1]  # End of each row in the data array
            row_sizes = row_ends - row_starts  # Number of nonzeros per row
            total_nonzeros = row_sizes.sum()

            # Preallocate memory for CSR arrays
            chunk_data = np.empty(total_nonzeros, dtype=data.dtype)
            chunk_indices = np.empty(total_nonzeros, dtype=indices.dtype)
            chunk_indptr = np.zeros(len(row_indices) + 1, dtype=np.int32)

            # Fill new arrays efficiently with NumPy slicing
            pos = 0
            for i, (start, end) in enumerate(zip(row_starts, row_ends)):
                num_values = end - start
                chunk_data[pos : pos + num_values] = data[start:end]
                chunk_indices[pos : pos + num_values] = indices[start:end]
                chunk_indptr[i + 1] = chunk_indptr[i] + num_values
                pos += num_values

        return csr_matrix(
            (chunk_data, chunk_indices, chunk_indptr), shape=(len(chunk_indptr) - 1, full_matrix_shape[1])
        )

    def _read_dict_chunk(self, group: h5py.Group, row_selector: slice | Iterable[int] = slice(None, None)):
        keys = group.keys()

        result = {}
        for key in keys:
            decoded_key = _decode_if_bytes(key)
            value = group[key]
            if isinstance(value, h5py.Group):
                value = self._read_group(value)
            elif isinstance(value, h5py.Dataset):
                value = self._read_dataset(value, row_selector)
            else:
                value = _decode_if_bytes(value[()])
            result[decoded_key] = value
        return result

    def _read_dataframe(
        self, group: h5py.Group, row_selector: slice | Iterable[int] = slice(None, None)
    ) -> pd.DataFrame:
        data = {}
        index_field = group.attrs.get("_index", "_index")

        for col_name in group.keys():
            # skipping reserved names
            if col_name.startswith("_"):
                continue

            col_group = group[col_name]
            if type(col_group) is h5py.Group and "categories" in col_group and "codes" in col_group:
                categories = col_group["categories"][:]
                codes = col_group["codes"][row_selector]

                if categories.dtype.kind in ("O", "S"):  # Fixed-length string dtype
                    categories = categories.astype(str)  # Convert to regular strings

                # Use pd.Categorical.from_codes for efficient categorical data handling
                data[col_name] = pd.Categorical.from_codes(codes, categories)
            else:
                data[col_name] = self._read_dataset(col_group, row_selector)

        df = pd.DataFrame(
            data,
            copy=False,
            index=group[index_field][row_selector] if type(group[index_field]) is h5py.Dataset else group[index_field],
        )

        # Ensure index and column names are strings
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)

        return df

    def _read_dataset(self, dataset: h5py.Dataset, row_selector: slice | Iterable[int] = slice(None, None)):
        if dataset.ndim == 0:
            result = dataset[()]
        else:
            if isinstance(row_selector, slice):
                result = dataset[row_selector]
            else:
                # datasets are read much faster using a slice , compared to a list of indices
                # therefore, we first read the whole slice into memory and than select specific indices
                # note that the number of indices in row_selector is limited by the max chunk size of the select_rows
                # method
                row_indices = np.asarray(row_selector, dtype=np.int32)
                df = dataset[row_indices[0] : row_indices[-1] + 1]
                result = df[row_indices - row_indices[0]]
        return _decode_if_bytes(result)


def _decode_if_bytes(item):
    if isinstance(item, bytes):
        return item.decode("utf-8")
    elif isinstance(item, np.ndarray):
        # Handle ndarrays with dtype=object that contain byte strings
        if item.dtype == object:
            return np.array([_decode_if_bytes(sub_item) for sub_item in item], dtype=object)
        elif item.dtype.type is np.bytes_:
            return item.astype(str)  # Efficiently convert byte arrays to strings
        else:
            return item  # Return the ndarray unchanged if it's not byte-encoded
    elif isinstance(item, list):
        # If it's a list, decode each item and return a list
        return [_decode_if_bytes(sub_item) for sub_item in item]
    elif isinstance(item, tuple):
        # If it's a tuple, decode each item but return a tuple
        return tuple(_decode_if_bytes(sub_item) for sub_item in item)
    else:
        return item  # If it's neither bytes nor an array, return as-is
