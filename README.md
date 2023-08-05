# rpz

Roman Photo-Z: Photometric Redshift Simulations for the Roman Space Telescope

[![Latest Release][release-badge]][release-url]
[![License][license-badge]](LICENSE)
[![CI Status][ci-badge]][ci-url]

[release-badge]: https://img.shields.io/github/v/release/austinlucaslake/rpz
[release-url]: https://github.com/austinlucaslake/rpz/releases/latest

[license-badge]: https://img.shields.io/github/license/austinlucaslake/rpz

[ci-badge]: https://github.com/austinlucaslake/rpz/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/austinlucaslake/rpz/actions

## Environment Setup

If you have just cloned the repository, please run the setup script provided:

`source setup.sh`

Aside from when you initially clone the repository, please use the following script upon opening a new shell to configure the environment:

`source env.sh`

## Modeling

To compute models using the provide code, please refer to `src/rpz/model.py`. This file contains functions to compute fluxes and magnitudes of observed spectra.
