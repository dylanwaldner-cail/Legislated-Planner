# Legislative-Harness

This repository contains research code for a multi-layer constraint harness for vision-language-action (VLA) models, built on top of two open source projects.

## Dependencies

**lerobot** (https://github.com/huggingface/lerobot) is a robotics library from Hugging Face that provides training and evaluation infrastructure for robot learning policies. We use it as the base evaluation framework and policy runner. Our modifications to lerobot are tracked in this repo under `lerobot/`.

**LIBERO** (https://github.com/Lifelong-Robot-Learning/LIBERO) is a benchmark for robot manipulation tasks. It provides the simulation environments, task definitions, and initial states we evaluate on. Our modifications to LIBERO are tracked in this repo under `libero/`.

## Setup

See SETUP.md for installation instructions, including known issues and fixes specific to this codebase.

To make the legislative harness importable from anywhere in the codebase (including inside lerobot and libero), install it as a package in your environment:

```bash
pip install -e .
```

Run this from the root of this repository after completing the setup steps in SETUP.md. You only need to do this once per environment.

## Changes

See CHANGES.md for an account of the location, nature, and motivations for the changes made.
