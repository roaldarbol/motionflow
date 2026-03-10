# _vendor

This directory contains vendored architecture code from third-party repositories
that are not available on PyPI. The files are downloaded on demand by running:

    python -m motionflow.methods._vendor_setup

## Contents

### sea_raft/
Vendored from: https://github.com/princeton-vl/SEA-RAFT
License: BSD-3-Clause (see sea_raft/LICENSE)
Files: core/ (model architecture only, no training/evaluation code)

### neuflow_v2/
Vendored from: https://github.com/neufieldrobotics/NeuFlow_v2
License: Apache-2.0 (see neuflow_v2/LICENSE)
Files: neuflow.py, backbone.py, refine.py (inference architecture only)

## Why vendoring?

Neither repo ships on PyPI and neither exposes a hubconf.py for torch.hub.
Vendoring the minimal architecture files is the cleanest way to support these
methods without requiring users to clone research repos or manage git dependencies.

## Updating vendored code

Re-run `python -m motionflow.methods._vendor_setup` to pull the latest upstream
versions. Pin a specific commit by editing _vendor_setup.py.
