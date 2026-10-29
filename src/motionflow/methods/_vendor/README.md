# Vendored model code

Model architectures that aren't on PyPI, copied into motionflow at pinned
commits so that installing motionflow is all that's needed. Each file starts
with a header naming its source and listing motionflow's changes, which are
kept to what's needed to import and run the models here.

| Folder | Upstream | Commit | Licence |
|---|---|---|---|
| `sea_raft/` | [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) (`core/`) | `9137517ba24e628442aec097d3afe71d03503b75` | BSD 3-Clause |
| `neuflow_v2/` | [neufieldrobotics/NeuFlow_v2](https://github.com/neufieldrobotics/NeuFlow_v2) (`NeuFlow/`) | `204b5e3744461d90303b9ff82caa7a1bb56a2ca2` | Apache 2.0 |

The weights are not vendored; motionflow downloads them on first use (see
`motionflow/weights.py`).

To update: copy the files from the new commit, reapply the changes listed in
each file's header, update the commit here, and run the deep-method tests.
