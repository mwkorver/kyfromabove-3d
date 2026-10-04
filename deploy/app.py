#!/usr/bin/env python3
"""The CDK app: ``cd deploy && npx aws-cdk deploy``. The region is
us-west-2, beside the KyFromAbove bucket."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aws_cdk as cdk                                                     # noqa: E402

from stack import BuildingStepStack, Ec2BenchStack, PieceStepStack, TileStepStack, ViewerStack  # noqa: E402

# step by step: steps 1, 2 and 3, each its own Lambda, invoked directly
# (PiecesStack, the event-driven chain, stays in stack.py for a later step)
app = cdk.App()
env = cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region="us-west-2")
TileStepStack(app, "kyfromabove-3d-tile-step", env=env)
PieceStepStack(app, "kyfromabove-3d-piece-step", env=env)
BuildingStepStack(app, "kyfromabove-3d-building-step", env=env)
# a Graviton Spot machine to price EC2 against Lambda; destroy it when done
Ec2BenchStack(app, "kyfromabove-3d-ec2-bench", env=env, count=int(app.node.try_get_context("bench_count") or 1),
              first_subnet=int(app.node.try_get_context("bench_subnet") or 0))
# the demo viewer: CloudFront over the served prefixes; the password (a
# git-ignored file) only with -c viewer_password=on
_pw = Path(__file__).with_name("viewer_password.txt")
ViewerStack(app, "kyfromabove-3d-viewer", env=env,
            password=_pw.read_text().strip() if app.node.try_get_context("viewer_password") == "on" and _pw.exists()
            else None)
app.synth()
