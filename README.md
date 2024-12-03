# softcopy

[![Release](https://img.shields.io/github/v/release/royerlab/softcopy)](https://img.shields.io/github/v/release/royerlab/softcopy)
[![Build status](https://img.shields.io/github/actions/workflow/status/royerlab/softcopy/main.yml?branch=main)](https://github.com/royerlab/softcopy/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/royerlab/softcopy/branch/main/graph/badge.svg)](https://codecov.io/gh/royerlab/softcopy)
[![Commit activity](https://img.shields.io/github/commit-activity/m/royerlab/softcopy)](https://img.shields.io/github/commit-activity/m/royerlab/softcopy)
[![License](https://img.shields.io/github/license/royerlab/softcopy)](https://img.shields.io/github/license/royerlab/softcopy)

Copies zarr archives from an acquisition frontend to a different disk, using filesystem watching and lockfiles to allow copying during acquisition.

> NOTE:
> This tool monitors lockfiles in order to detect changes from tensorstore. You must set file_io_locking.mode to `lockfile`
> if you want to be sure this will work with your writer!
> https://google.github.io/tensorstore/kvstore/file/index.html#json-Context.file_io_locking.mode

## Background
At the Royer Lab, we have several microscopes which perform large (many TB) acquisitions over a long (~24 hour) period, using a Zarr archive to compress and store bioimages on disk.
Once the acquisition finishes, we then move the data from the acquisition frontend (the PC in the room, ~100TB storage) to our high performance compute cluster. This allows us to
free up storage on the frontend and lets our scientists perform analysis on the HPC.

These datasets are so large, however, that copying data to the HPC can take days, even over connections that are considered fast in the consumer market (>GbPS). This means that
the instrument drives will be full for a long time, and limits how often we can acquire datasets.

Softcopy was built to address this issue. It is able to copy zarr archives file-by-file, while they are being written to, with high throughput but low disk and cpu priority.
This allows much of the data copying to happen during acquisition - the acquisition control software writes chunks with tensorstore, and softcopy starts copying it to cold storage
immediately.

Microscope frontends are usually heavily IO constrained - on a machine with spinning disks, streaming from HD cameras can easily reach 100% disk utilization. Softcopy aims
to use the disk and CPU as little as possible to prevent putting too much additional strain on system resources. It does this by monitoring filesystem events rather than polling
the disk, knowing what files to expect from the zarr format, and using OS IO priority controls and queues to allow the disk to feed in data only when the time is right.

Softcopy is only designed to work with `tensorstore` - `tensorstore` is the fastest zarr writer we are aware of, which is crucial for our applications - but it also uses
lockfiles which enable softcopy to identify which files are not ready to be copied.

## Installation
Use `pip install softcopy` to install softcopy globally on your system. For a less permanent installation, use a venv or install [uv](https://docs.astral.sh/uv/) and run `uvx softcopy`.

## Usage
For basic usage instructions, see `softcopy --help`:

```
$ softcopy --help
Usage: softcopy [OPTIONS] TARGETS_FILE

  Tranfer data from source to destination as described in a yaml TARGETS_FILE.
  Uses low priority io to allow data to be moved while the microscope is
  acquiring. The program is zarr-aware and can safely copy an archive before
  it is finished being written to.

Options:
  --verbose                  print debug information while running
  --nprocs INTEGER           number of processes to use for copying
  --sleep-time FLOAT         time to sleep in each copy process between
                             copies. Can help mitigate down an overwhelemd
                             system
  --wait-for-source BOOLEAN  If the source does not exist when softcopy is
                             started, wait for it to appear. If false,
                             softcopy will crash if the source does not exist
  --help                     Show this message and exit.
```

### --verbose
Shows initialization steps, input zarr analysis, certain filesystem events and other debug info.

### --nprocs INTEGER
The number of processes which each copy operation will spawn. Each process will copy one file at a time, which in theory allows the os IO scheduling to feed
processes more efficiently from the read heads of spinning disks (and helps saturate network io). Note that this is per-copy - if you are copying 4 zarr archives
from your targets file, and set `--nprocs 3`, `softcopy` will spawn `3 * 4 = 12` subprocesses. As a rule of thumb, set this to 4 or 8 and only adjust if softcopy
is causing performance degredation even after using a small sleep time.

### --sleep-time FLOAT
Each writer process is effectively running the following pseudocode:
```ruby
until no_more_files_to_copy do
  copy_next_file
  sleep_this_thread(sleep_time)
end
```
By default the sleep time is zero - so softcopy will go as fast as possible. If softcopy is slowing down your acquisition process by hogging the disk, this is
the most important thing to change. Setting a small, modest sleep time (could vary wildly depending on network, zarr chunk size) such as 0.5 will likely help
a lot and keep softcopy from starving the disk.

## Target file format
We use softcopy to copy from multiple in-progress zarr archives at the same time. Specifying many long filepaths on different disks is tedious and confusing, so softcopy
accepts a description of its task via a yaml file (commonly called `targets.yaml`). Currently, this is simply an enumeration of "copy from" and "copy to" paths:

```yaml
# The toplevel key for everything is called `targets`. It is a list of dicts, and must contain at least one source/destination pair.
targets:
  # The first dict here corresponds to the first copy operation
  - source: /path/to/in-progress/acquisition/data.zarr
    destination: /path/to/server/where/softcopy/should/move/the/data.zarr
  # The subsequent dicts (if any) will all describe other copies. Everything will happen in parallel, order doesn't matter
  - source: /path/to/other/source.zarr
    destination: /path/to/other/destination.zarr
  # ...
```

In future versions of softcopy, other settings may be added on a per-target basis.

## Pitfalls
Softcopy is still in development so I haven't written this section yet. However note that there may be issues with:

### Writing chunks in multiple shots
As an example of this, if you have a chunk shape of 10x10 (i.e. each file in the zarr archive contains a 10x10 array),
and your writer writes to a chunk partially at time 1, then waits a bit, and finishes filling that chunk at time 2, then softcopy
will probably copy that file multiple times. More concretely:

```python
dataset = ts.open(...).result()

dataset[0, :].write(row1).result()
# This transaction finishes, and will dispatch an fs write event for the partial chunk (chunk id 0,0). softcopy sees this and copies it


dataset[1, :].write(row2).result()
# This transaction finishes, which will send another fs event for the SAME chunk (0,0). Softcopy will then copy this file again...
```

This did not really occur to me during development - for disk utilization, we have always written full chunk files (usually `dataset[timepoint, ...].write(entire czyx stack).result()`).
However, if you do not have enough RAM to store an entire czyx stack, you will have to be careful to make your partial stack writes align neatly along chunk boundaries to prevent
chunks from being written to non-atomically.

I should also note that I have not tested this - and I have no idea how softcopy would handle it. My instinct tells me that softcopy will just copy the chunk over and over, wasting bandwidth but still getting the job done in the end - but there could easily be a concurrency bug or race condition here that results in the partial chunk overwriting the full chunk.

### Checksums
Softcopy does not currently do any impressive integrity checking (yet). Even if hashing was instantaneous, just loading a 100TB acquisition through memory would take 3 hours on an 8GBps NVMe drive. After the copy finishes, softcopy *does* do a basic integrity check, though - it knows what files a zarr archive of the given shape should contain, and it makes sure the destination has every file that is expected. I highly reccommend checking the filesizes of the source and destination archive to ensure you aren't missing data (at the very least - I'm still figuring out how to check for corruption).
