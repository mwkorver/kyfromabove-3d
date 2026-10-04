"""The stage-1 stack, synthesised without AWS or Docker: the EventBridge rule
matches exactly a piece's marker, it feeds a queue with a dead-letter queue,
the Lambdas are sized for a tile, and the bucket announces its objects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

cdk = pytest.importorskip("aws_cdk")
from aws_cdk import assertions                                              # noqa: E402
from aws_cdk import aws_lambda as lambda_                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

from stack import PiecesStack                                               # noqa: E402


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    tiny = tmp_path_factory.mktemp("image")
    (tiny / "Dockerfile").write_text("FROM public.ecr.aws/lambda/python:3.12\n")
    app = cdk.App()
    stack = PiecesStack(app, "Test", keep_data=False,
                        image=lambda h: lambda_.DockerImageCode.from_image_asset(str(tiny), cmd=[h]))
    return assertions.Template.from_stack(stack)


def test_the_bucket_sends_its_events_to_eventbridge(template):
    template.has_resource_properties("Custom::S3BucketNotifications", {
        "NotificationConfiguration": {"EventBridgeConfiguration": {}}})


def test_the_rule_matches_only_a_pieces_marker_and_feeds_a_queue(template):
    rules = template.find_resources("AWS::Events::Rule")
    assert len(rules) == 1
    rule = next(iter(rules.values()))["Properties"]
    pattern = rule["EventPattern"]
    assert pattern["source"] == ["aws.s3"] and pattern["detail-type"] == ["Object Created"]
    assert pattern["detail"]["object"]["key"] == [{"wildcard": "pieces/*/piece.json"}]
    assert "Fn::GetAtt" in rule["Targets"][0]["Arn"]                        # the pieces queue
    queues = template.find_resources("AWS::SQS::Queue")
    assert sum("RedrivePolicy" in q["Properties"] for q in queues.values()) == 2   # both queues have a DLQ


def test_the_lambdas_are_sized_for_their_work(template):
    template.has_resource_properties("AWS::Lambda::Function", {
        "MemorySize": 6144, "Timeout": 900, "EphemeralStorage": {"Size": 10240}, "Architectures": ["arm64"],
        "ImageConfig": {"Command": ["handlers.make_pieces"]}})
    template.has_resource_properties("AWS::Lambda::Function", {
        "MemorySize": 512, "ImageConfig": {"Command": ["handlers.record_piece"]}})
    template.has_resource_properties("AWS::Lambda::EventSourceMapping", {
        "BatchSize": 1, "FunctionResponseTypes": ["ReportBatchItemFailures"], "ScalingConfig": {"MaximumConcurrency": 50}})


def test_the_pieces_table_is_keyed_by_building_and_indexed_by_tile(template):
    template.has_resource_properties("AWS::DynamoDB::Table", {
        "KeySchema": [{"AttributeName": "gers", "KeyType": "HASH"}],
        "GlobalSecondaryIndexes": [assertions.Match.object_like({"IndexName": "by_tile"})]})


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


def test_step_two_is_one_named_lambda_on_the_existing_bucket_and_nothing_else(tmp_path_factory):
    from stack import PieceStepStack

    tiny = tmp_path_factory.mktemp("image2")
    (tiny / "Dockerfile").write_text("FROM public.ecr.aws/lambda/python:3.12\n")
    t = assertions.Template.from_stack(PieceStepStack(
        cdk.App(), "Step2", image=lambda h: lambda_.DockerImageCode.from_image_asset(str(tiny), cmd=[h])))
    t.resource_count_is("AWS::Lambda::Function", 1)
    t.has_resource_properties("AWS::Lambda::Function", {
        "FunctionName": "kyfromabove-3d-piece-step", "MemorySize": 2048, "Timeout": 900,
        "Architectures": ["arm64"], "ImageConfig": {"Command": ["handlers.piece_step"]}})
    t.has_resource_properties("AWS::IAM::Role", {"RoleName": "kyfromabove-3d-piece-step-role"})
    for kind in ("AWS::S3::Bucket", "AWS::SQS::Queue", "AWS::Events::Rule", "AWS::DynamoDB::Table"):
        t.resource_count_is(kind, 0)


def test_step_three_is_one_named_lambda_on_the_existing_bucket_and_nothing_else(tmp_path_factory):
    from stack import BuildingStepStack

    tiny = tmp_path_factory.mktemp("image3")
    (tiny / "Dockerfile").write_text("FROM public.ecr.aws/lambda/python:3.12\n")
    t = assertions.Template.from_stack(BuildingStepStack(
        cdk.App(), "Step3", image=lambda h: lambda_.DockerImageCode.from_image_asset(str(tiny), cmd=[h])))
    t.resource_count_is("AWS::Lambda::Function", 1)
    t.has_resource_properties("AWS::Lambda::Function", {
        "FunctionName": "kyfromabove-3d-building-step", "MemorySize": 3008, "Timeout": 900,
        "Architectures": ["arm64"], "EphemeralStorage": {"Size": 10240},
        "ImageConfig": {"Command": ["handlers.building_step"]}})
    t.has_resource_properties("AWS::IAM::Role", {"RoleName": "kyfromabove-3d-building-step-role"})
    for kind in ("AWS::S3::Bucket", "AWS::SQS::Queue", "AWS::Events::Rule", "AWS::DynamoDB::Table"):
        t.resource_count_is(kind, 0)


def test_the_ec2_bench_is_one_spot_machine_that_ends_itself_and_takes_no_inbound(tmp_path_factory):
    from aws_cdk import aws_ec2 as ec2
    from aws_cdk.aws_ecr_assets import DockerImageAsset
    from stack import BENCH_LIFETIME_MIN, Ec2BenchStack

    tiny = tmp_path_factory.mktemp("image4")
    (tiny / "Dockerfile").write_text("FROM public.ecr.aws/lambda/python:3.12\n")
    st = Ec2BenchStack(cdk.App(), "Bench",
                       vpc_factory=lambda s: ec2.Vpc(s, "Vpc", max_azs=1, nat_gateways=0),
                       image_factory=lambda s: DockerImageAsset(s, "Image", directory=str(tiny)))
    t = assertions.Template.from_stack(st)
    t.resource_count_is("AWS::EC2::Instance", 1)
    t.has_resource_properties("AWS::EC2::LaunchTemplate", {"LaunchTemplateData": assertions.Match.object_like({
        "InstanceType": "c8g.4xlarge", "InstanceInitiatedShutdownBehavior": "terminate",
        "InstanceMarketOptions": {"MarketType": "spot", "SpotOptions": assertions.Match.object_like(
            {"MaxPrice": "0.35", "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"})}})})
    sgs = t.find_resources("AWS::EC2::SecurityGroup")
    assert all(not r["Properties"].get("SecurityGroupIngress") for r in sgs.values())
    t.resource_count_is("AWS::EC2::SecurityGroupIngress", 0)
    lt = next(iter(t.find_resources("AWS::EC2::LaunchTemplate").values()))
    ud = json.dumps(lt["Properties"]["LaunchTemplateData"]["UserData"])
    assert f"shutdown -h +{BENCH_LIFETIME_MIN}" in ud
    assert "proxy_cache_path /var/cache/nginx/cog" in ud and "slice" in ud       # the frame cache's config
    t.has_resource_properties("AWS::IAM::Role", {"RoleName": "kyfromabove-3d-ec2-bench-role"})


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
