#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TurboQuant-patched vLLM serve entrypoint.

The sitecustomize.py import hook patches CacheDType at module load time,
before EngineArgs builds argparse choices. This script applies the
remaining runtime patches (backend routing, attention buffers, etc)
then delegates to vLLM's CLI.
"""

import sys
sys.path.insert(0, "/opt")
sys.path.insert(0, "/llms/turboquant-xpu-tests")

# Apply remaining TurboQuant patches
import turboquant_register  # noqa: F401

# Delegate to vLLM CLI
from vllm.entrypoints.cli.main import main
main()
