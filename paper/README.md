# paper/

This directory should contain the exact manuscript artifact that matches the public repository.

## Recommended contents

- `manuscript.pdf`: the exact PDF that the repository is supposed to support;
- `source.zip` or `arxiv.tex` plus any required source tree;
- `figures/` if you want paper assets versioned in the repo;
- optional `tables/` symlink or copies if the paper consumes generated LaTeX snippets from the root `tables/` directory.

## Why this matters

If the root README says that the repository can generate or populate the paper tables, the public repository should also tell a reader where the paper source and final PDF live.
