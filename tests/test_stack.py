"""The stacks, synthesised without AWS or Docker: the queue run on Spot,
the viewer and the budget."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

cdk = pytest.importorskip("aws_cdk")
from aws_cdk import assertions                                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

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
    assert _run_gate(open_code, "/viewer/vcmap/")["uri"] == "/viewer/vcmap/index.html"   # folder address
    moved = _run_gate(open_code, "/viewer/vcmap")
    assert moved["statusCode"] == 302 and moved["headers"]["location"]["value"] == "/viewer/vcmap/"
    assert _run_gate(open_code, "/viewer/vcmap/assets/ui.js")["uri"] == "/viewer/vcmap/assets/ui.js"
    assert _run_gate(open_code, "/buildings/")["statusCode"] == 403
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
    # the data bucket is imported and its own policy stays as it is; the one
    # bucket made here holds the access log, kept 30 days
    t.resource_count_is("AWS::S3::Bucket", 1)
    t.has_resource_properties("AWS::S3::Bucket", assertions.Match.object_like({
        "BucketName": "kyfromabove-3d-viewer-logs",
        "LifecycleConfiguration": {"Rules": [assertions.Match.object_like({"ExpirationInDays": 30})]}}))
    t.has_resource_properties("AWS::CloudFront::Distribution", {"DistributionConfig": assertions.Match.object_like({
        "Logging": assertions.Match.object_like({"Prefix": "cloudfront/"})})})
    logs = next(iter(t.find_resources("AWS::S3::Bucket")))
    for p in t.find_resources("AWS::S3::BucketPolicy").values():                 # none on the data bucket
        assert p["Properties"]["Bucket"] == {"Ref": logs}
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


def test_the_budgets_count_the_project_and_viewer_tags_and_warn_before_and_at_the_limit():
    from stack import BudgetStack

    t = assertions.Template.from_stack(BudgetStack(cdk.App(), "Budget", email="someone@example.com", limit_usd=700,
                                                   viewer_usd=5))
    t.resource_count_is("AWS::Budgets::Budget", 2)
    for usd, tag in ((700, "user:project$kyfromabove-3d"), (5, "user:component$viewer")):
        t.has_resource_properties("AWS::Budgets::Budget", {
            "Budget": assertions.Match.object_like({
                "BudgetType": "COST", "TimeUnit": "MONTHLY", "BudgetLimit": {"Amount": usd, "Unit": "USD"},
                "CostFilters": {"TagKeyValue": [tag]}}),
            "NotificationsWithSubscribers": [
                {"Notification": assertions.Match.object_like({"NotificationType": "ACTUAL", "Threshold": 80}),
                 "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "someone@example.com"}]},
                {"Notification": assertions.Match.object_like({"NotificationType": "FORECASTED", "Threshold": 100}),
                 "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "someone@example.com"}]}]})


def test_the_viewer_alarms_watch_its_hourly_bytes_and_requests_and_email():
    from stack import ViewerAlarmStack

    t = assertions.Template.from_stack(ViewerAlarmStack(cdk.App(), "Alarms", distribution_id="E123",
                                                        email="someone@example.com", gb_per_hour=20))
    t.has_resource_properties("AWS::SNS::Subscription", {"Protocol": "email", "Endpoint": "someone@example.com"})
    t.resource_count_is("AWS::CloudWatch::Alarm", 2)
    for metric, limit in (("BytesDownloaded", 20e9), ("Requests", 1_000_000)):
        t.has_resource_properties("AWS::CloudWatch::Alarm", assertions.Match.object_like({
            "Namespace": "AWS/CloudFront", "MetricName": metric, "Statistic": "Sum", "Period": 3600,
            "Threshold": limit, "Dimensions": assertions.Match.array_with([{"Name": "DistributionId", "Value": "E123"}]),
            "AlarmActions": assertions.Match.any_value()}))


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
    from stack import QUEUE_INSTANCES, QUEUE_LIFETIME_MIN, QUEUE_WORKER_HOURS, QueueRunStack

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
                "SpotAllocationStrategy": "capacity-optimized-prioritized"})})})
    asg = next(iter(t.find_resources("AWS::AutoScaling::AutoScalingGroup").values()))["Properties"]
    types = [o["InstanceType"] for o in asg["MixedInstancesPolicy"]["LaunchTemplate"]["Overrides"]]
    assert types == list(QUEUE_INSTANCES) and types[0] == "c8g.4xlarge"      # Graviton4 first, in order
    lt = next(iter(t.find_resources("AWS::EC2::LaunchTemplate").values()))
    assert lt["Properties"]["LaunchTemplateData"]["MetadataOptions"]["HttpTokens"] == "required"
    ud = json.dumps(lt["Properties"]["LaunchTemplateData"]["UserData"])
    assert f"shutdown -h +{QUEUE_LIFETIME_MIN}" in ud and "scripts/queue_worker.py" in ud and "--run r1" in ud
    # a machine whose worker has stopped shuts down (terminates), after its log is saved
    assert ud.index("scripts/queue_worker.py") < ud.index("worker.log s3://") < ud.rindex("shutdown -h now")
    t.resource_count_is("AWS::EC2::SecurityGroupIngress", 0)
    # the worker: 2-hour jobs, and it stops taking jobs well before the machine's end
    assert "--timeout-s 7200" in ud and f"--max-hours {QUEUE_WORKER_HOURS}" in ud
    assert QUEUE_WORKER_HOURS * 60 + 120 < QUEUE_LIFETIME_MIN
    # machines in proportion to the queue (14 jobs a machine), not all of them for one job
    steps = next(iter(t.find_resources("AWS::AutoScaling::ScalingPolicy",
                                       {"Properties": {"PolicyType": "StepScaling"}}).values()))
    adj = sorted((a.get("MetricIntervalLowerBound", -1), a["ScalingAdjustment"])
                 for p in t.find_resources("AWS::AutoScaling::ScalingPolicy").values()
                 for a in p["Properties"].get("StepAdjustments", []))
    assert [m for _, m in adj][-3:] == [1, 2, 3] and steps
    # busy machines may protect themselves from scale-in
    t.has_resource_properties("AWS::IAM::Policy", {"PolicyDocument": {"Statement": assertions.Match.array_with([
        assertions.Match.object_like({"Action": ["autoscaling:DescribeAutoScalingInstances",
                                                  "autoscaling:SetInstanceProtection"]})])}})
    # a deploy that changes the template replaces the running machines
    t.has_resource("AWS::AutoScaling::AutoScalingGroup", {"UpdatePolicy": {"AutoScalingRollingUpdate": assertions.Match.object_like(
        {"MinInstancesInService": 0, "MaxBatchSize": 3})}})
