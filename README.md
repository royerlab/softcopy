# softcopy

[![Release](https://img.shields.io/github/v/release/royerlab/softcopy)](https://img.shields.io/github/v/release/royerlab/softcopy)
[![Build status](https://img.shields.io/github/actions/workflow/status/royerlab/softcopy/main.yml?branch=main)](https://github.com/royerlab/softcopy/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/royerlab/softcopy/branch/main/graph/badge.svg)](https://codecov.io/gh/royerlab/softcopy)
[![Commit activity](https://img.shields.io/github/commit-activity/m/royerlab/softcopy)](https://img.shields.io/github/commit-activity/m/royerlab/softcopy)
[![License](https://img.shields.io/github/license/royerlab/softcopy)](https://img.shields.io/github/license/royerlab/softcopy)

Copies zarr archives from an acquisition frontend to a different disk, using filesystem watching and lockfiles to allow copying during acquisition.

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

TODO: Document targets.yaml, how to use the CLI, pitfalls, etc.
