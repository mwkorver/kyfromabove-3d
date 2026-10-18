"""The stacks, synthesised without AWS or Docker: the tile step's Lambda,
the queue run on Spot, the viewer and the budget."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

cdk = pytest.importorskip("aws_cdk")
from aws_cdk import assertions                                              # noqa: E402
from aws_cdk import aws_lambda as lambda_                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

def test_step_one_is_one_named_lambda_on_the_existing_bucket_and_nothing_else(tmp_path_factory):
    from stack import TileStepStack

    tiny = tmp_path_factory.mktemp("image1")
    (tiny / "Dockerfile").write_text("FROM public.ecr.aws/lambda/python:3.12\n")
    t = assertions.Template.from_stack(TileStepStack(
        cdk.App(), "Step1", image=lambda h: lambda_.DockerImageCode.from_image_asset(str(tiny), cmd=[h])))
    t.resource_count_is("AWS::Lambda::Function", 1)
    t.has_resource_properties("AWS::Lambda::Function", {
        "FunctionName": "kyfromabove-3d-tile-step", "MemorySize": 3008, "Timeout": 900,
        "Architectures": ["arm64"], "ImageConfig": {"Command": ["handlers.tile_step"]},
        "Environment": {"Variables": {"BUCKET": "kyfromabove-3d", "RAW_DIR": "/opt/data/raw"}}})
    t.has_resource_properties("AWS::IAM::Role", {"RoleName": "kyfromabove-3d-tile-step-role"})
    t.has_resource_properties("AWS::Logs::LogGroup", {"LogGroupName": "/aws/lambda/kyfromabove-3d-tile-step",
                                                      "RetentionInDays": 14})
    for kind in ("AWS::S3::Bucket", "AWS::SQS::Queue", "AWS::Events::Rule", "AWS::DynamoDB::Table"):
        t.resource_count_is(kind, 0)                      # the bucket is imported; no queue, rule or table


def _run_gate(code: str, uri: str, auth: str | None = None) -> dict:
    """The CloudFront Function's own code, run under node on one request."""
    import shutil
    import subprocess
    if not shutil.which("node"):
        pytest.skip("node not installed")
    ev = {"request": {"uri": uri, "headers": {"authorization": {"value": auth}} if auth else {}}}
    js = code + f"\nprocess.stdout.write(JSON.stringify(handler({json.dumps(ev)})));"
    return json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)


def test_the_viewer_gate_serves_only_the_served_paths_and_asks_for_the_password():
    import base64
    from stack import VIEWER_USER, viewer_function_code

    open_code = viewer_function_code(None)
    assert _run_gate(open_code, "/tiles/N076E229/3dtiles/measured/cells/ab12.glb")["uri"].endswith("ab12.glb")
    assert _run_gate(open_code, "/kentucky/complete/tileset.json")["uri"] == "/kentucky/complete/tileset.json"
    assert _run_gate(open_code, "/terrain/15/1/2.terrain")["uri"].startswith("/terrain/")
    assert _run_gate(open_code, "/buildings/g1/audit.json")["statusCode"] == 403          # private
    assert _run_gate(open_code, "/pieces/g1/frames.json")["statusCode"] == 403
    assert _run_gate(open_code, "/tiles/N076E229/buildings.json")["statusCode"] == 403   # not under 3dtiles/
    assert _run_gate(open_code, "/")["statusCode"] == 302
    locked = viewer_function_code("s3cret-pw")
    assert _run_gate(locked, "/viewer/index.html")["statusCode"] == 401
    assert _run_gate(locked, "/viewer/index.html", "Basic " + base64.b64encode(b"kentucky:wrong").decode())["statusCode"] == 401
    ok = "Basic " + base64.b64encode(f"{VIEWER_USER}:s3cret-pw".encode()).decode()
    assert _run_gate(locked, "/viewer/index.html", ok)["uri"] == "/viewer/index.html"
    assert _run_gate(locked, "/buildings/g1/audit.json", ok)["statusCode"] == 403         # the password opens no more


def test_the_viewer_stack_leaves_the_bucket_policy_and_prints_the_read_statement():
    from stack import SERVED_PREFIXES, ViewerStack

    t = assertions.Template.from_stack(ViewerStack(cdk.App(), "Viewer", password="pw"))
    t.resource_count_is("AWS::CloudFront::Distribution", 1)
    t.resource_count_is("AWS::S3::Bucket", 0)                                    # the bucket is imported
    t.resource_count_is("AWS::S3::BucketPolicy", 0)                              # its own policy stays as it is
    out = t.find_outputs("BucketPolicyStatement")
    text = json.dumps(next(iter(out.values()))["Value"])
    assert "ViewerReadsServedPrefixes" in text and "s3:GetObject" in text and "cloudfront.amazonaws.com" in text
    assert "AWS:SourceArn" in text                                               # this distribution only
    for p in SERVED_PREFIXES:
        assert f"kyfromabove-3d/{p}" in text
    fn = next(iter(t.find_resources("AWS::CloudFront::Function").values()))
    assert "401" in json.dumps(fn["Properties"]["FunctionCode"])


def test_the_viewer_policy_adds_its_statement_and_keeps_the_rest():
    from viewer_policy import merged

    https = {"Sid": "HttpsOnly", "Effect": "Deny"}
    read = {"Sid": "ViewerReadsServedPrefixes", "Effect": "Allow", "Resource": ["new"]}
    once = merged({"Version": "2012-10-17", "Statement": [https]}, read)
    assert once["Statement"] == [https, read]
    again = merged(once, dict(read, Resource=["newer"]))                         # a redeploy replaces, never duplicates
    assert again["Statement"] == [https, dict(read, Resource=["newer"])]


def test_the_budget_counts_the_project_tag_and_warns_before_and_at_the_limit():
    from stack import BudgetStack

    t = assertions.Template.from_stack(BudgetStack(cdk.App(), "Budget", email="someone@example.com", limit_usd=700))
    t.resource_count_is("AWS::Budgets::Budget", 1)
    t.has_resource_properties("AWS::Budgets::Budget", {
        "Budget": assertions.Match.object_like({
            "BudgetType": "COST", "TimeUnit": "MONTHLY", "BudgetLimit": {"Amount": 700, "Unit": "USD"},
            "CostFilters": {"TagKeyValue": ["user:project$kyfromabove-3d"]}}),
        "NotificationsWithSubscribers": [
            {"Notification": assertions.Match.object_like({"NotificationType": "ACTUAL", "Threshold": 80}),
             "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "someone@example.com"}]},
            {"Notification": assertions.Match.object_like({"NotificationType": "FORECASTED", "Threshold": 100}),
             "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "someone@example.com"}]}]})


def test_the_project_tag_reaches_the_spot_machines_and_their_disks(tmp_path_factory):
    """The tag is set once on the app (deploy/app.py); the Spot machines are
    the bill's largest part, so they and their disks must carry it."""
    from aws_cdk import aws_ec2 as ec2
    from aws_cdk.aws_ecr_assets import DockerImageAsset
    from stack import PROJECT_TAG, QueueRunStack

    tiny = tmp_path_factory.mktemp("image5")
    (tiny / "Dockerfile").write_text("FROM public.ecr.aws/lambda/python:3.12\n")
    app = cdk.App()
    st = QueueRunStack(app, "Queue", run="r1", vpc_factory=lambda s: ec2.Vpc(s, "Vpc", max_azs=2, nat_gateways=0),
                       image_factory=lambda s: DockerImageAsset(s, "Image", directory=str(tiny)))
    cdk.Tags.of(app).add(*PROJECT_TAG)
    t = assertions.Template.from_stack(st)
    t.has_resource_properties("AWS::AutoScaling::AutoScalingGroup", {"Tags": assertions.Match.array_with([
        {"Key": PROJECT_TAG[0], "Value": PROJECT_TAG[1], "PropagateAtLaunch": True}])})
    tag = {"Key": PROJECT_TAG[0], "Value": PROJECT_TAG[1]}
    lt = next(iter(t.find_resources("AWS::EC2::LaunchTemplate").values()))["Properties"]["LaunchTemplateData"]
    tagged = {s["ResourceType"] for s in lt.get("TagSpecifications", []) if tag in s["Tags"]}
    assert {"instance", "volume"} <= tagged, tagged


def test_the_queue_run_is_spot_machines_on_a_queue_with_a_patient_dead_letter_queue(tmp_path_factory):
    from aws_cdk import aws_ec2 as ec2
    from aws_cdk.aws_ecr_assets import DockerImageAsset
    from stack import QUEUE_INSTANCES, QUEUE_LIFETIME_MIN, QueueRunStack

    tiny = tmp_path_factory.mktemp("image6")
    (tiny / "Dockerfile").write_text("FROM public.ecr.aws/lambda/python:3.12\n")
    st = QueueRunStack(cdk.App(), "Queue", run="r1", max_machines=3,
                       vpc_factory=lambda s: ec2.Vpc(s, "Vpc", max_azs=2, nat_gateways=0),
                       image_factory=lambda s: DockerImageAsset(s, "Image", directory=str(tiny)))
    t = assertions.Template.from_stack(st)
    t.has_resource_properties("AWS::SQS::Queue", {"QueueName": "kyfromabove-3d-jobs", "VisibilityTimeout": 900,
                                                  "RedrivePolicy": assertions.Match.object_like({"maxReceiveCount": 10})})
    t.has_resource_properties("AWS::AutoScaling::AutoScalingGroup", {
        "MinSize": "0", "MaxSize": "3", "MixedInstancesPolicy": assertions.Match.object_like({
            "InstancesDistribution": assertions.Match.object_like({
                "OnDemandBaseCapacity": 0, "OnDemandPercentageAboveBaseCapacity": 0,
                "SpotAllocationStrategy": "capacity-optimized"})})})
    asg = next(iter(t.find_resources("AWS::AutoScaling::AutoScalingGroup").values()))["Properties"]
    types = [o["InstanceType"] for o in asg["MixedInstancesPolicy"]["LaunchTemplate"]["Overrides"]]
    assert types == list(QUEUE_INSTANCES)
    lt = next(iter(t.find_resources("AWS::EC2::LaunchTemplate").values()))
    ud = json.dumps(lt["Properties"]["LaunchTemplateData"]["UserData"])
    assert f"shutdown -h +{QUEUE_LIFETIME_MIN}" in ud and "scripts/queue_worker.py" in ud and "--run r1" in ud
    # a machine whose worker has stopped shuts down (terminates), after its log is saved
    assert ud.index("scripts/queue_worker.py") < ud.index("worker.log s3://") < ud.rindex("shutdown -h now")
    t.resource_count_is("AWS::EC2::SecurityGroupIngress", 0)
    # a deploy that changes the template replaces the running machines
    t.has_resource("AWS::AutoScaling::AutoScalingGroup", {"UpdatePolicy": {"AutoScalingRollingUpdate": assertions.Match.object_like(
        {"MinInstancesInService": 0, "MaxBatchSize": 3})}})
