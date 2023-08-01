# roman_photo_z

Photometric redshift simulations for NASA's Roman Space Telescope

[![Latest Release][release-badge]][release-url]
[![License][license-badge]](LICENSE)
[![CI Status][ci-badge]][ci-url]

[release-badge]: https://img.shields.io/github/v/release/austinlucaslake/roman_photo_z
[release-url]: https://github.com/austinlucaslake/roman_photo_z/releases/latest

[license-badge]: https://img.shields.io/github/license/austinlucaslake/roman_photo_z

[ci-badge]: https://github.com/austinlucaslake/roman_photo_z/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/austinlucaslake/roman_photo_z/actions

## Environment Setup

If you have just cloned the repository, please run the setup script provided:

`source setup.sh`

Aside from when you initially clone the repository, please use the following script upon opening a new shell to configure the environment:

`source env.sh`

## Modeling

To compute models using the provide code, please refer to `src/roman_photo_z/model.py`. This file contains functions to compute fluxes and magnitudes of observed spectra.
