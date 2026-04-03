from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from hypersphere_dpp.hdf5_utils import open_readonly_h5


class OpenReadonlyH5Test(unittest.TestCase):
    def test_retries_oserror_lock_failures(self):
        lock_error = OSError(
            "Unable to synchronously open file (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable')"
        )
        expected_handle = object()

        with (
            mock.patch(
                "hypersphere_dpp.hdf5_utils.h5py.File",
                side_effect=[lock_error, expected_handle],
            ) as mocked_h5py_file,
            mock.patch("hypersphere_dpp.hdf5_utils.time.sleep") as mocked_sleep,
        ):
            handle = open_readonly_h5(Path("train-00000-of-00001.h5"), attempts=2, sleep_seconds=0.01)

        self.assertIs(handle, expected_handle)
        self.assertEqual(mocked_h5py_file.call_count, 2)
        mocked_sleep.assert_called_once_with(0.01)

    def test_falls_back_when_locking_kw_is_unsupported(self):
        expected_handle = object()

        with (
            mock.patch(
                "hypersphere_dpp.hdf5_utils.h5py.File",
                side_effect=TypeError("unexpected keyword argument 'locking'"),
            ) as mocked_h5py_file,
            mock.patch(
                "hypersphere_dpp.hdf5_utils._open_readonly_h5_without_locking_kw",
                return_value=expected_handle,
            ) as mocked_fallback,
        ):
            handle = open_readonly_h5(Path("train-00000-of-00001.h5"), attempts=1)

        self.assertIs(handle, expected_handle)
        mocked_h5py_file.assert_called_once()
        mocked_fallback.assert_called_once_with(Path("train-00000-of-00001.h5"), swmr=True)


if __name__ == "__main__":
    unittest.main()
