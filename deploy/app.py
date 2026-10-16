#!/usr/bin/env python3
"""The CDK app: ``cd deploy && npx aws-cdk deploy``. The region is
us-west-2, beside the KyFromAbove bucket."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aws_cdk as cdk                                                     # noqa: E402

from stack import PROJECT_TAG, BudgetStack, QueueRunStack, TileStepStack, ViewerStack  # noqa: E402

app = cdk.App()
env = cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region="us-west-2")
# step 1, each tile's inputs, one Lambda invoked per tile (deploy/run_tile_step.py)
TileStepStack(app, "kyfromabove-3d-tile-step", env=env)
# a run on Spot machines fed by SQS (refactor step 4), only when a run is named:
# npx aws-cdk deploy kyfromabove-3d-queue -c queue_run=r1 [-c queue_machines=2]
if app.node.try_get_context("queue_run"):
    QueueRunStack(app, "kyfromabove-3d-queue", env=env, run=app.node.try_get_context("queue_run"),
                  max_machines=int(app.node.try_get_context("queue_machines") or 2))
# the demo viewer: CloudFront over the served prefixes; the password (a
# git-ignored file) only with -c viewer_password=on
_pw = Path(__file__).with_name("viewer_password.txt")
ViewerStack(app, "kyfromabove-3d-viewer", env=env,
            password=_pw.read_text().strip() if app.node.try_get_context("viewer_password") == "on" and _pw.exists()
            else None)
# a monthly budget on the project's tag, only when an address is given:
# npx aws-cdk deploy kyfromabove-3d-budget -c budget_email=you@example.com [-c budget_usd=700]
if app.node.try_get_context("budget_email"):
    BudgetStack(app, "kyfromabove-3d-budget", env=env, email=app.node.try_get_context("budget_email"),
                limit_usd=float(app.node.try_get_context("budget_usd") or 20))
# every stack, and what it makes, carries the project's tag (Billing can then
# split the bill by it, and the budget counts it)
cdk.Tags.of(app).add(*PROJECT_TAG)
app.synth()
