from __future__ import annotations

import os
import time
from pathlib import Path

# Some HDF5 builds still consult this environment variable when deciding
# whether to attempt POSIX file locks during readonly opens.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py

_LOCK_ERROR_MESSAGE_TOKENS = (
    "unable to lock file",
    "file locking",
    "resource temporarily unavailable",
    "no locks available",
    "errno = 11",
    "errno = 35",
    "errno = 37",
)


def is_hdf5_lock_error(error: BaseException) -> bool:
    if isinstance(error, BlockingIOError):
        return True
    message = str(error).lower()
    return any(token in message for token in _LOCK_ERROR_MESSAGE_TOKENS)


def _open_readonly_h5_low_level(path: Path) -> h5py.File:
    if not hasattr(h5py.h5p.PropFAID, "set_file_locking"):
        raise AttributeError("h5py low-level file locking controls are unavailable.")

    file_access_property_list = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    try:
        file_access_property_list.set_file_locking(False, True)
    except TypeError:
        file_access_property_list.set_file_locking(False)

    file_id = h5py.h5f.open(
        os.fsencode(path),
        h5py.h5f.ACC_RDONLY,
        fapl=file_access_property_list,
    )
    return h5py.File(file_id)


def _open_readonly_h5_without_locking_kw(path: Path, *, swmr: bool) -> h5py.File:
    try:
        return _open_readonly_h5_low_level(path)
    except Exception:
        pass

    try:
        return h5py.File(path, "r", swmr=swmr)
    except TypeError:
        return h5py.File(path, "r")


def open_readonly_h5(
    path: str | Path,
    *,
    attempts: int = 20,
    sleep_seconds: float = 0.5,
    swmr: bool = True,
) -> h5py.File:
    resolved_path = Path(path)
    last_error: BaseException | None = None

    for attempt in range(attempts):
        try:
            return h5py.File(resolved_path, "r", locking=False, swmr=swmr)
        except TypeError:
            try:
                return _open_readonly_h5_without_locking_kw(resolved_path, swmr=swmr)
            except (BlockingIOError, OSError) as error:
                if not is_hdf5_lock_error(error):
                    raise
                last_error = error
        except (BlockingIOError, OSError) as error:
            if not is_hdf5_lock_error(error):
                raise
            last_error = error

        if attempt + 1 >= attempts:
            break
        time.sleep(sleep_seconds)

    raise BlockingIOError(
        f"Unable to open readonly HDF5 file after {attempts} attempts due to file locking: {resolved_path}"
    ) from last_error
