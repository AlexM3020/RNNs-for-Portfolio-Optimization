# Consolidation record

## Source versions

| Repository | Source commit |
|---|---|
| AlexM3020/portfolio-lstm | 180889d1dff70d84b5496125d191ec3d1d23fe0b |
| AlexM3020/PortfolioOptimizationICM | 3993334223923ffee2c585278a5bdedf82918ff7 |

Consolidation was prepared on **refactor/cv-consolidation**. On 2026-11-11, the project author confirmed the paper and authorized publication to **portfolio-lstm/main**. The secondary repository remains untouched; repository retirement is a separate decision.

## Included

- 12 original files from portfolio-lstm: 11 scripts + original README, under [legacy/code-repository](../legacy/code-repository/).
- 46 substantive files from PortfolioOptimizationICM: 13 scripts + README, 4 data snapshots, 15 numeric artifacts, and 14 figures.
- Every original substantive file retains its original checkout bytes. Git source-blob SHA-256 values are also recorded; CRLF checkout differences are explicit.
- Duplicate numeric exports and older data variants remain in the historical archive, not in the active pipeline.
- [migration_manifest.json](migration_manifest.json) records source paths, destination paths, fingerprints, and exclusions.

## Excluded from the curated project

- Six .idea configuration files: machine/editor metadata.
- The `code` Git entry: an unresolved gitlink at commit 258be3896b3b254c87ed3da5ad24ab95cbf26fe8. No .gitmodules URL was committed; the checkout contains no content there. This link is preserved in the original repository’s complete Git backup, not fabricated into source files.
- Git internals, local virtual environments, generated runs, and uncommitted artifacts that were never available from GitHub.

No source repository was cleaned or deleted.

## History backups

Complete Git bundles for **both repositories** were created in the sibling folder **portfolio-repository-backups**, outside this project. The source bundle was verified with Git. These backups preserve history, not merely the currently visible files. Keep them somewhere backed up before deleting a GitHub repository; the bundles are intentionally not added to the CV project.

## Integrity check

From the project root, run `python scripts/verify_migration.py`. This checks all 58 migrated file hashes without needing the secondary checkout or GitHub access. Archive files are marked non-text/binary in Git attributes where appropriate, preventing Windows line-ending conversion from invalidating their fingerprints after publication.

## Before retiring the secondary repository

1. Review and publish the consolidated branch to portfolio-lstm.
2. Check the README, image links, CI, and a fresh clone’s migration verification.
3. Retain both complete Git backups separately.
4. Consider **archiving** PortfolioOptimizationICM rather than deleting it if its old URLs are already shared. Deletion would break those URLs; consolidation does not redirect them.

The current local migration contains all substantive tracked source files. Remote publication and any repository retirement remain separate decisions.