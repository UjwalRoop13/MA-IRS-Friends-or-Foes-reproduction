# Movable Antennas Meet Intelligent Reflecting Surface: Friends or Foes?

Independent Python implementation and reproduction of the numerical experiments
presented in:

> X. Wei, W. Mei, Q. Wu, Q. Jia, B. Ning, Z. Chen, and J. Fang,
> "Movable Antennas Meet Intelligent Reflecting Surface: Friends or Foes?",
> IEEE Transactions on Communications, vol. 73, no. 11, pp. 12756–12770, 2025.

## Overview

This repository contains an independent Python implementation of the optimization
algorithms, channel models, Monte-Carlo simulations, and numerical experiments
presented in the paper.

The implementation focuses on an IRS-assisted multi-user MISO system with
Movable Antennas (MAs) at the base station.

The reproduced experiments include:

- Fig. 2 – Received SNR at the user
- Fig. 3 – Single-user achievable rate
- Fig. 6 – Multi-user sum-rate with LoS BS-IRS channel
- Fig. 8 – Multi-user sum-rate with multi-path BS-IRS channel

## Main Components

The implementation includes:

- NUSW-based LoS BS-IRS channel model
- Rician IRS-user channel
- IRS passive beamforming
- Block Coordinate Descent (BCD)
- Dynamic programming / graph-based antenna-position selection
- WMMSE beamforming
- Complex-circle manifold optimization
- Discrete-sampling antenna-position optimization
- Alternating optimization (AO)
- Monte-Carlo simulation
- Numerical validation checks

## System Model

The considered system consists of:

- A multi-antenna BS equipped with movable antennas
- An intelligent reflecting surface (IRS)
- Multiple single-antenna users
- Negligible direct BS-user channels

The BS jointly optimizes:

1. Active beamforming
2. IRS passive beamforming
3. Movable antenna positions

subject to transmit-power, unit-modulus, and minimum antenna-spacing
constraints.

## Implementation Notes

This is an independent reproduction rather than an official implementation
from the paper authors.

Some information required to reproduce the simulations exactly is not explicitly
specified in the paper. Therefore, additional assumptions are documented in the
source code.

Examples include:

- BS position/range handling
- Movable-region orientation
- Multi-path scatterer locations
- User directions
- Fig. 2 IRS-user distance
- Algorithm 3 initialization

The code also documents several equation/model inconsistencies identified during
implementation and applies explicit corrections where necessary.

## Reproduction Results

### Fig. 2 – Received SNR

![Fig. 2](figures/fig2_received_snr.png)

### Fig. 3 – Single-user achievable rate

![Fig. 3](figures/fig3_single_user_rate.png)

### Fig. 6 – Sum-rate, LoS BS-IRS channel

#### Algorithm 3 – Best initialization

![Fig. 6 Best](figures/fig6_sum_rate_best.png)

#### Algorithm 3 – Compact initialization

![Fig. 6 Compact](figures/fig6_sum_rate_compact.png)

### Fig. 8 – Sum-rate, multi-path BS-IRS channel

#### Algorithm 3 – Best initialization

![Fig. 8 Best](figures/fig8_sum_rate_best.png)

#### Algorithm 3 – Compact initialization

![Fig. 8 Compact](figures/fig8_sum_rate_compact.png)

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ma-irs-friends-or-foes-reproduction.git
cd ma-irs-friends-or-foes-reproduction
