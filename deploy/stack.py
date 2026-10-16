"""The AWS stacks. What is deployed and runs: one Lambda each for the tile
and piece steps (``TileStepStack``, ``PieceStepStack``), the building step's
Lambda used for calibration (``BuildingStepStack``), the Spot machines the
building step runs on (``Ec2BenchStack``), the CloudFront viewer
(``ViewerStack``) and the monthly budget (``BudgetStack``). A laptop's
runners (``deploy/run_*.py``) invoke each step directly.

Designed, tested against moto and never deployed: ``PiecesStack``, an event
chain that would join the steps without the laptop --

    tiles queue --> make_pieces (Lambda, container) --> S3 pieces/<gers>/..., piece.json last
                                                         |
                          S3 "Object Created" --> EventBridge rule (pieces/*/piece.json)
                                                         |
                                pieces queue --> record_piece (Lambda) --> DynamoDB "pieces"

Each queue has a dead-letter queue: a tile or a piece that fails twice is
parked there rather than retried forever. The queues also cap how many
Lambdas run at once (``max_concurrency``), which EventBridge straight to
Lambda would not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from aws_cdk import CfnOutput, Duration, Fn, RemovalPolicy, Size, Stack
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_lambda_event_sources as sources
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sqs as sqs
from constructs import Construct

ROOT = Path(__file__).resolve().parents[1]

#: The EventBridge pattern for a finished piece: its marker, last to be written.
PIECE_PATTERN_KEY = [{"wildcard": "pieces/*/piece.json"}]

#: What the image build leaves out of the repository: outputs, the local
#: environment, and every raw file but the tile grid the image copies.
IMAGE_EXCLUDE = ["outputs", ".venv", ".git", "notebooks", "docs", "tests", "tools", "viewers", "cdk.out",
                 "deploy/cdk.out", "data/derived", "data/raw/photos", "data/raw/*.laz",
                 "data/raw/KY_*", "data/raw/kyfromabove_phase1_*"]


#: Graviton: about 20% cheaper than x86 for the same work, and every library
#: the pipeline uses publishes ARM wheels.
ARCH = lambda_.Architecture.ARM_64


def repo_image(handler: str) -> lambda_.DockerImageCode:
    """The pipeline's image (deploy/Dockerfile), running ``handler``."""
    from aws_cdk.aws_ecr_assets import Platform

    return lambda_.DockerImageCode.from_image_asset(str(ROOT), file="deploy/Dockerfile", cmd=[handler],
                                                    exclude=IMAGE_EXCLUDE, platform=Platform.LINUX_ARM64)


def _step_function(stack: Stack, cid: str, function_name: str, handler: str, memory_mb: int, tmp_gb: int,
                   description: str, bucket_name: str, image) -> lambda_.Function:
    """One named Lambda from the pipeline's image, with its own role and log
    group, reading and writing the project's bucket."""
    from aws_cdk import aws_iam as iam
    from aws_cdk import aws_logs as logs

    bucket = s3.Bucket.from_bucket_name(stack, "Data", bucket_name)
    role = iam.Role(stack, "Role", role_name=f"{function_name}-role",
                    assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
                    managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name(
                        "service-role/AWSLambdaBasicExecutionRole")])
    log_group = logs.LogGroup(stack, "Logs", log_group_name=f"/aws/lambda/{function_name}",
                              retention=logs.RetentionDays.TWO_WEEKS, removal_policy=RemovalPolicy.DESTROY)
    fn = lambda_.DockerImageFunction(
        stack, cid, function_name=function_name, code=(image or repo_image)(handler),
        architecture=ARCH, memory_size=memory_mb, timeout=Duration.minutes(15),
        ephemeral_storage_size=Size.gibibytes(tmp_gb), role=role, log_group=log_group, description=description,
        environment={"BUCKET": bucket_name, "RAW_DIR": "/opt/data/raw"})
    bucket.grant_read_write(fn)
    CfnOutput(stack, "FunctionName", value=fn.function_name)
    CfnOutput(stack, "BucketName", value=bucket_name)
    return fn


class PieceStepStack(Stack):
    """Step 2 alone: one Lambda, invoked directly once per batch of a tile's
    buildings, writing each building's piece to the project's bucket."""

    FUNCTION_NAME = "kyfromabove-3d-piece-step"

    def __init__(self, scope: Construct, cid: str, *, bucket_name: str = "kyfromabove-3d", memory_mb: int = 2048,
                 image: Optional[Callable[[str], lambda_.DockerImageCode]] = None, **kw):
        super().__init__(scope, cid, **kw)
        self.function = _step_function(
            self, "PieceStep", self.FUNCTION_NAME, "handlers.piece_step", memory_mb, 2,
            "Step 2: a batch of one tile's buildings cut into pieces (LiDAR clip, frames as pointers)",
            bucket_name, image)


class BuildingStepStack(Stack):
    """Step 3 alone: one Lambda, invoked directly with a few nearby
    buildings, reading their pieces and writing their meshes, check and
    audit to ``buildings/<version>/<gers>/``. No queue, no event rule."""

    FUNCTION_NAME = "kyfromabove-3d-building-step"

    def __init__(self, scope: Construct, cid: str, *, bucket_name: str = "kyfromabove-3d",
                 image: Optional[Callable[[str], lambda_.DockerImageCode]] = None, **kw):
        super().__init__(scope, cid, **kw)
        # 3008 MB, the account's ceiling (and about two vCPUs); 10 GB of /tmp
        # for a tower's crops, its clip and the LiDAR around it
        self.function = _step_function(
            self, "BuildingStep", self.FUNCTION_NAME, "handlers.building_step", 3008, 10,
            "Step 3: buildings from their pieces to meshes (measured and completed), texture check and audit",
            bucket_name, image)


class TileStepStack(Stack):
    """Step 1 alone: one Lambda, invoked directly once per LiDAR tile,
    writing to the project's existing bucket. No queue, no event rule."""

    FUNCTION_NAME = "kyfromabove-3d-tile-step"

    def __init__(self, scope: Construct, cid: str, *, bucket_name: str = "kyfromabove-3d",
                 image: Optional[Callable[[str], lambda_.DockerImageCode]] = None, **kw):
        super().__init__(scope, cid, **kw)
        # 3008 MB: this account's ceiling for a Lambda's memory (new accounts
        # get 3008 until a quota increase); the tile step peaked at 1.6 GB
        self.function = _step_function(
            self, "TileStep", self.FUNCTION_NAME, "handlers.tile_step", 3008, 10,
            "Step 1: one LiDAR tile's Overture buildings measured (product 1) and its frames listed",
            bucket_name, image)


class PiecesStack(Stack):
    """The event chain in the module's docstring. Not deployed: the pilot
    invoked each step directly. Kept, with its tests, for a statewide run."""

    def __init__(self, scope: Construct, cid: str, *, tile_concurrency: int = 50, keep_data: bool = True,
                 image: Optional[Callable[[str], lambda_.DockerImageCode]] = None, **kw):
        super().__init__(scope, cid, **kw)

        bucket = s3.Bucket(
            self, "Data", event_bridge_enabled=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL, encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN if keep_data else RemovalPolicy.DESTROY,
            auto_delete_objects=not keep_data)

        table = ddb.Table(
            self, "Pieces", partition_key=ddb.Attribute(name="gers", type=ddb.AttributeType.STRING),
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN if keep_data else RemovalPolicy.DESTROY)
        table.add_global_secondary_index(index_name="by_tile",
                                         partition_key=ddb.Attribute(name="tile", type=ddb.AttributeType.STRING))

        image = image or repo_image

        # stage 1: a tile a message, one at a time per Lambda
        tiles_dlq = sqs.Queue(self, "TilesDLQ", retention_period=Duration.days(14))
        tiles = sqs.Queue(self, "Tiles", visibility_timeout=Duration.minutes(90),
                          dead_letter_queue=sqs.DeadLetterQueue(queue=tiles_dlq, max_receive_count=2))
        make = lambda_.DockerImageFunction(
            self, "MakePieces", code=image("handlers.make_pieces"), architecture=ARCH, memory_size=6144,
            timeout=Duration.minutes(15), ephemeral_storage_size=Size.gibibytes(10),
            environment={"BUCKET": bucket.bucket_name, "RAW_DIR": "/opt/data/raw"})
        bucket.grant_read_write(make)
        make.add_event_source(sources.SqsEventSource(tiles, batch_size=1, report_batch_item_failures=True,
                                                     max_concurrency=tile_concurrency))

        # a piece's marker lands -> EventBridge -> the pieces queue
        pieces_dlq = sqs.Queue(self, "PiecesDLQ", retention_period=Duration.days(14))
        pieces = sqs.Queue(self, "PiecesReady", visibility_timeout=Duration.minutes(6),
                           dead_letter_queue=sqs.DeadLetterQueue(queue=pieces_dlq, max_receive_count=2))
        events.Rule(self, "PieceArrived", event_pattern=events.EventPattern(
            source=["aws.s3"], detail_type=["Object Created"],
            detail={"bucket": {"name": [bucket.bucket_name]}, "object": {"key": PIECE_PATTERN_KEY}}),
            targets=[targets.SqsQueue(pieces)])

        # the next step, this round: record it
        record = lambda_.DockerImageFunction(
            self, "RecordPiece", code=image("handlers.record_piece"), architecture=ARCH, memory_size=512,
            timeout=Duration.minutes(1), environment={"TABLE": table.table_name})
        bucket.grant_read(record)
        table.grant_write_data(record)
        record.add_event_source(sources.SqsEventSource(pieces, batch_size=10, report_batch_item_failures=True,
                                                       max_concurrency=5))

        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "TilesQueueUrl", value=tiles.queue_url)
        CfnOutput(self, "PiecesTable", value=table.table_name)
        self.bucket, self.table, self.tiles, self.pieces = bucket, table, tiles, pieces



#: The Spot bench's machine: Graviton4, 16 processors, 32 GB.
BENCH_INSTANCE = "c8g.4xlarge"
#: Never billed above this an hour (Spot), and never alive past this: a
#: forgotten machine costs at most BENCH_MAX_PRICE x BENCH_LIFETIME_MIN / 60.
BENCH_MAX_PRICE = "0.35"
BENCH_LIFETIME_MIN = 300
#: nginx for the frame cache, from ECR Public (no Docker Hub pull limits)
COG_CACHE_NGINX = "public.ecr.aws/nginx/nginx:1.27-alpine"


class Ec2BenchStack(Stack):
    """One Graviton Spot machine to time the building step on, beside the
    Lambda: the same image, run with ``scripts/building_step.py``. Reached
    through SSM (no SSH, no inbound port); reads and writes the project's
    bucket; stops itself after ``BENCH_LIFETIME_MIN`` minutes and is then
    terminated. Destroy the stack when done."""

    def __init__(self, scope: Construct, cid: str, *, bucket_name: str = "kyfromabove-3d",
                 vpc_factory: Optional[Callable[[Stack], object]] = None,
                 image_factory: Optional[Callable[[Stack], object]] = None, count: int = 1,
                 first_subnet: int = 0, **kw):
        """``count`` machines, one per public subnet (zone) in turn from
        ``first_subnet``: a zone out of Spot capacity is skipped by moving it."""
        super().__init__(scope, cid, **kw)
        from aws_cdk import aws_ec2 as ec2
        from aws_cdk import aws_iam as iam

        vpc = (vpc_factory or (lambda s: ec2.Vpc.from_lookup(s, "Default", is_default=True)))(self)
        if image_factory is None:
            from aws_cdk.aws_ecr_assets import DockerImageAsset, Platform
            image = DockerImageAsset(self, "Image", directory=str(ROOT), file="deploy/Dockerfile",
                                     exclude=IMAGE_EXCLUDE, platform=Platform.LINUX_ARM64)
        else:
            image = image_factory(self)

        role = iam.Role(self, "Role", role_name="kyfromabove-3d-ec2-bench-role",
                        assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
                        managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name(
                            "AmazonSSMManagedInstanceCore")])
        s3.Bucket.from_bucket_name(self, "Data", bucket_name).grant_read_write(role)
        image.repository.grant_pull(role)
        sg = ec2.SecurityGroup(self, "NoInbound", vpc=vpc, allow_all_outbound=True,
                               description="kyfromabove-3d bench: no inbound; out to S3, ECR and KyFromAbove")

        ud = ec2.UserData.for_linux()
        ud.add_commands(
            f"shutdown -h +{BENCH_LIFETIME_MIN}",               # first: the machine's hard end
            "dnf install -y docker && systemctl start docker",
            f"aws ecr get-login-password --region {self.region} | docker login --username AWS "
            f"--password-stdin {self.account}.dkr.ecr.{self.region}.amazonaws.com",
            f"docker pull {image.image_uri} && docker tag {image.image_uri} building-step:bench",
            # the frame cache's config (deploy/cog_cache), and nginx, ready to start
            "mkdir -p /opt/cog_cache /var/tmp/cog_log",
            "cat > /opt/cog_cache/nginx.conf.template <<'KFA_NGINX'",
            (ROOT / "deploy" / "cog_cache" / "nginx.conf.template").read_text().rstrip("\n"),
            "KFA_NGINX",
            f"docker pull {COG_CACHE_NGINX}",
            "touch /var/tmp/bench-ready")
        lt = ec2.LaunchTemplate(
            self, "Template", instance_type=ec2.InstanceType(BENCH_INSTANCE),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(cpu_type=ec2.AmazonLinuxCpuType.ARM_64),
            role=role, security_group=sg, user_data=ud,
            instance_initiated_shutdown_behavior=ec2.InstanceInitiatedShutdownBehavior.TERMINATE,
            spot_options=ec2.LaunchTemplateSpotOptions(
                max_price=float(BENCH_MAX_PRICE), request_type=ec2.SpotRequestType.ONE_TIME,
                interruption_behavior=ec2.SpotInstanceInterruption.TERMINATE),
            block_devices=[ec2.BlockDevice(device_name="/dev/xvda", volume=ec2.BlockDeviceVolume.ebs(
                40, volume_type=ec2.EbsDeviceVolumeType.GP3, delete_on_termination=True))])
        ids = []
        for i in range(count):
            inst = ec2.CfnInstance(self, "Bench" if i == 0 else f"Bench{i}",
                                   subnet_id=vpc.public_subnets[(first_subnet + i) % len(vpc.public_subnets)].subnet_id,
                                   launch_template=ec2.CfnInstance.LaunchTemplateSpecificationProperty(
                                       launch_template_id=lt.launch_template_id,
                                       version=lt.latest_version_number))
            ids.append(inst.ref)
        CfnOutput(self, "InstanceId", value=ids[0])
        CfnOutput(self, "InstanceIds", value=",".join(ids) if count == 1 else Fn.join(",", ids))
        CfnOutput(self, "ImageUri", value=image.image_uri)


#: The queue run's machines: Graviton, 16 processors, 32-64 GB, whichever a
#: zone has Spot capacity for (capacity-optimized).
QUEUE_INSTANCES = ("c8g.4xlarge", "c7g.4xlarge", "m8g.4xlarge", "m7g.4xlarge")
QUEUE_MAX_PRICE = "0.45"
#: A machine's hard end: however the run goes, none lives past this.
QUEUE_LIFETIME_MIN = 720


class QueueRunStack(Stack):
    """A run on EC2 Spot fed by SQS (refactor step 4): the job queue (a
    dead-letter queue behind it at ``max_receives``, high because a Spot
    interruption also counts as a receive -- the worker parks a job after its
    own count of real failures), and an Auto Scaling group of Graviton Spot
    machines that each run ``scripts/queue_worker.py`` in the pipeline's
    image, beside the frame cache. The group grows to ``max_machines`` while
    the queue holds work (visible or in flight) and shrinks to none when it is
    empty. Reached through SSM only; no inbound port. Each machine stops
    itself after ``QUEUE_LIFETIME_MIN`` minutes."""

    def __init__(self, scope: Construct, cid: str, *, run: str = "run", bucket_name: str = "kyfromabove-3d",
                 max_machines: int = 2, slots: int = 14, max_receives: int = 10,
                 vpc_factory: Optional[Callable[[Stack], object]] = None,
                 image_factory: Optional[Callable[[Stack], object]] = None, **kw):
        super().__init__(scope, cid, **kw)
        from aws_cdk import aws_autoscaling as autoscaling
        from aws_cdk import aws_cloudwatch as cw
        from aws_cdk import aws_ec2 as ec2
        from aws_cdk import aws_iam as iam

        vpc = (vpc_factory or (lambda s: ec2.Vpc.from_lookup(s, "Default", is_default=True)))(self)
        if image_factory is None:
            from aws_cdk.aws_ecr_assets import DockerImageAsset, Platform
            image = DockerImageAsset(self, "Image", directory=str(ROOT), file="deploy/Dockerfile",
                                     exclude=IMAGE_EXCLUDE, platform=Platform.LINUX_ARM64)
        else:
            image = image_factory(self)

        dlq = sqs.Queue(self, "Parked", queue_name="kyfromabove-3d-jobs-dlq", retention_period=Duration.days(14))
        queue = sqs.Queue(self, "Jobs", queue_name="kyfromabove-3d-jobs", visibility_timeout=Duration.minutes(15),
                          retention_period=Duration.days(14),
                          dead_letter_queue=sqs.DeadLetterQueue(queue=dlq, max_receive_count=max_receives))

        role = iam.Role(self, "Role", role_name="kyfromabove-3d-queue-role",
                        assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
                        managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name(
                            "AmazonSSMManagedInstanceCore")])
        s3.Bucket.from_bucket_name(self, "Data", bucket_name).grant_read_write(role)
        queue.grant_consume_messages(role)
        image.repository.grant_pull(role)
        sg = ec2.SecurityGroup(self, "NoInbound", vpc=vpc, allow_all_outbound=True,
                               description="kyfromabove-3d queue run: no inbound; out to S3, SQS, ECR and KyFromAbove")

        ud = ec2.UserData.for_linux()
        worker = (f"docker run --name worker --network host --entrypoint python -w /var/task "
                  f"-e OMP_NUM_THREADS=1 -e COG_CACHE_URL=http://127.0.0.1:8001 -e STORE=s3://{bucket_name} "
                  f"-e WORK_DIR=/var/tmp/work -e AWS_REGION={self.region} -v /var/tmp/work:/var/tmp/work "
                  f"{image.image_uri} scripts/queue_worker.py --queue-url {queue.queue_url} --run {run} "
                  f"--slots {slots} --idle-exit-s 1800 > /var/tmp/worker.log 2>&1")
        ud.add_commands(
            f"shutdown -h +{QUEUE_LIFETIME_MIN}",               # first: the machine's hard end
            "dnf install -y docker && systemctl start docker",
            f"aws ecr get-login-password --region {self.region} | docker login --username AWS "
            f"--password-stdin {self.account}.dkr.ecr.{self.region}.amazonaws.com",
            f"docker pull {image.image_uri}",
            "mkdir -p /opt/cog_cache /var/tmp/cog_log /var/tmp/work",
            "cat > /opt/cog_cache/nginx.conf.template <<'KFA_NGINX'",
            (ROOT / "deploy" / "cog_cache" / "nginx.conf.template").read_text().rstrip("\n"),
            "KFA_NGINX",
            f"docker run -d --name kfa-cog-cache --network host -e SLICE=1m -e CACHE_MAX=20g "
            f"-v /opt/cog_cache/nginx.conf.template:/etc/nginx/templates/cog.conf.template:ro "
            f"-v kfa-cog-cache:/var/cache/nginx/cog -v /var/tmp/cog_log:/var/log/nginx {COG_CACHE_NGINX}",
            worker,
            "aws s3 cp /var/tmp/worker.log "
            f"s3://{bucket_name}/runs/{run}/logs/$(cat /var/lib/cloud/data/instance-id).log || true")
        lt = ec2.LaunchTemplate(
            self, "Template", machine_image=ec2.MachineImage.latest_amazon_linux2023(
                cpu_type=ec2.AmazonLinuxCpuType.ARM_64),
            role=role, security_group=sg, user_data=ud,
            instance_initiated_shutdown_behavior=ec2.InstanceInitiatedShutdownBehavior.TERMINATE,
            block_devices=[ec2.BlockDevice(device_name="/dev/xvda", volume=ec2.BlockDeviceVolume.ebs(
                60, volume_type=ec2.EbsDeviceVolumeType.GP3, delete_on_termination=True))])
        asg = autoscaling.AutoScalingGroup(
            self, "Machines", vpc=vpc, vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            min_capacity=0, max_capacity=max_machines,
            mixed_instances_policy=autoscaling.MixedInstancesPolicy(
                launch_template=lt,
                launch_template_overrides=[autoscaling.LaunchTemplateOverrides(instance_type=ec2.InstanceType(t))
                                           for t in QUEUE_INSTANCES],
                instances_distribution=autoscaling.InstancesDistribution(
                    on_demand_base_capacity=0, on_demand_percentage_above_base_capacity=0,
                    spot_allocation_strategy=autoscaling.SpotAllocationStrategy.CAPACITY_OPTIMIZED,
                    spot_max_price=QUEUE_MAX_PRICE)))
        # all the machines while the queue holds work, none when it is empty; work in flight
        # counts (a job being built is not visible, and its machine must stay)
        work = cw.MathExpression(expression="visible + flight", period=Duration.minutes(1), using_metrics={
            "visible": queue.metric_approximate_number_of_messages_visible(period=Duration.minutes(1)),
            "flight": queue.metric_approximate_number_of_messages_not_visible(period=Duration.minutes(1))})
        asg.scale_on_metric("ByQueue", metric=work, adjustment_type=autoscaling.AdjustmentType.EXACT_CAPACITY,
                            scaling_steps=[autoscaling.ScalingInterval(upper=0, change=0),
                                           autoscaling.ScalingInterval(lower=1, change=max_machines)],
                            evaluation_periods=5)
        CfnOutput(self, "QueueUrl", value=queue.queue_url)
        CfnOutput(self, "DeadLetterQueueUrl", value=dlq.queue_url)
        CfnOutput(self, "GroupName", value=asg.auto_scaling_group_name)
        CfnOutput(self, "ImageUri", value=image.image_uri)


#: What the viewer's CloudFront may read from the bucket: the page, the
#: statewide tree, each tile's batched tiles and the terrain. Everything else
#: (pieces, building outputs, audits, logs) stays private even to CloudFront.
SERVED_PREFIXES = ("viewer/*", "kentucky/*", "tiles/*/3dtiles/*", "terrain/*")
VIEWER_USER = "kentucky"


def viewer_function_code(password: str | None) -> str:
    """The CloudFront Function on every request: the password (HTTP basic
    auth, user ``kentucky``) when one is given, then only the served paths;
    ``/`` goes to the viewer."""
    import base64

    check = ""
    if password:
        token = base64.b64encode(f"{VIEWER_USER}:{password}".encode()).decode()
        check = f"""
  var auth = req.headers.authorization;
  if (!auth || auth.value !== "Basic {token}") {{
    return {{statusCode: 401, statusDescription: "Unauthorized",
            headers: {{"www-authenticate": {{value: 'Basic realm="Kentucky buildings demo"'}}}}}};
  }}"""
    return f"""function handler(event) {{
  var req = event.request;{check}
  var uri = req.uri;
  if (uri === "/" || uri === "/viewer" || uri === "/viewer/") {{
    return {{statusCode: 302, statusDescription: "Found", headers: {{location: {{value: "/viewer/index.html"}}}}}};
  }}
  if (!/^\\/(viewer|kentucky|terrain)\\//.test(uri) && !/^\\/tiles\\/[^\\/]+\\/3dtiles\\//.test(uri)) {{
    return {{statusCode: 403, statusDescription: "Forbidden"}};
  }}
  return req;
}}"""


class ViewerStack(Stack):
    """The demo viewer: CloudFront in front of the project's private bucket
    through origin access control, serving only ``SERVED_PREFIXES``, behind
    one shared password (a CloudFront Function). Long caching on the
    content-named tiles and terrain; five minutes on JSON, which is rewritten
    in place. The bucket's policy is managed here: its HTTPS-only statement
    as it was, plus CloudFront's read access to the served prefixes."""

    def __init__(self, scope: Construct, cid: str, *, bucket_name: str = "kyfromabove-3d",
                 password: str | None = None, **kw):
        super().__init__(scope, cid, **kw)
        from aws_cdk import aws_cloudfront as cf
        from aws_cdk import aws_cloudfront_origins as origins
        from aws_cdk import aws_iam as iam

        bucket = s3.Bucket.from_bucket_name(self, "Data", bucket_name)
        fn = cf.Function(self, "Gate", function_name="kyfromabove-3d-viewer-gate",
                         runtime=cf.FunctionRuntime.JS_2_0,
                         code=cf.FunctionCode.from_inline(viewer_function_code(password)),
                         comment="password and served paths for the demo viewer")
        gate = [cf.FunctionAssociation(function=fn, event_type=cf.FunctionEventType.VIEWER_REQUEST)]
        origin = origins.S3BucketOrigin.with_origin_access_control(bucket)
        short = cf.CachePolicy(self, "Json", cache_policy_name="kyfromabove-3d-viewer-json",
                               default_ttl=Duration.minutes(5), max_ttl=Duration.minutes(5), min_ttl=Duration.seconds(0),
                               enable_accept_encoding_gzip=True, enable_accept_encoding_brotli=True)
        dist = cf.Distribution(
            self, "Viewer", comment="kyfromabove-3d demo viewer", price_class=cf.PriceClass.PRICE_CLASS_100,
            default_behavior=cf.BehaviorOptions(
                origin=origin, viewer_protocol_policy=cf.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cf.CachePolicy.CACHING_OPTIMIZED, compress=True, function_associations=gate),
            additional_behaviors={"*.json": cf.BehaviorOptions(
                origin=origin, viewer_protocol_policy=cf.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=short, compress=True, function_associations=gate)})
        # The bucket was made by hand and has its own policy (HTTPS only), which
        # CloudFormation cannot take over without first deleting it. So the
        # stack leaves the policy alone and prints the one statement CloudFront
        # needs -- reads of the served prefixes by this distribution only -- to
        # be added next to the existing ones (deploy/viewer_policy.py).
        read = iam.PolicyStatement(sid="ViewerReadsServedPrefixes", effect=iam.Effect.ALLOW,
                                   principals=[iam.ServicePrincipal("cloudfront.amazonaws.com")],
                                   actions=["s3:GetObject"],
                                   resources=[f"{bucket.bucket_arn}/{p}" for p in SERVED_PREFIXES],
                                   conditions={"StringEquals": {"AWS:SourceArn":
                                               f"arn:aws:cloudfront::{self.account}:distribution/{dist.distribution_id}"}})
        CfnOutput(self, "BucketPolicyStatement", value=Fn.to_json_string(read.to_statement_json()))
        CfnOutput(self, "ViewerUrl", value=f"https://{dist.distribution_domain_name}/viewer/index.html")
        CfnOutput(self, "PasswordOn", value="yes" if password else "no")


#: Every stack is tagged with this (deploy/app.py), and the budget counts
#: only what carries it. AWS adds "user:" to tags people set.
PROJECT_TAG = ("project", "kyfromabove-3d")


class BudgetStack(Stack):
    """A monthly AWS Budget on what carries the project's tag: an email at 80%
    of ``limit_usd`` actually spent, and another when the month is forecast
    to pass it. It warns; it does not stop anything.

    Two things it cannot do for itself, once per account: the bucket was made
    by hand, so it needs the tag added to it (its storage and requests are a
    large share of the bill), and the tag must be switched on as a cost
    allocation tag in Billing before a budget can filter on it."""

    def __init__(self, scope: Construct, cid: str, *, email: str, limit_usd: float = 20.0, **kw):
        super().__init__(scope, cid, **kw)
        from aws_cdk import aws_budgets as budgets

        key, value = PROJECT_TAG

        def alert(kind: str, percent: float):
            return budgets.CfnBudget.NotificationWithSubscribersProperty(
                notification=budgets.CfnBudget.NotificationProperty(
                    notification_type=kind, comparison_operator="GREATER_THAN", threshold=percent,
                    threshold_type="PERCENTAGE"),
                subscribers=[budgets.CfnBudget.SubscriberProperty(subscription_type="EMAIL", address=email)])

        budgets.CfnBudget(self, "Monthly", budget=budgets.CfnBudget.BudgetDataProperty(
            budget_name="kyfromabove-3d-monthly", budget_type="COST", time_unit="MONTHLY",
            budget_limit=budgets.CfnBudget.SpendProperty(amount=limit_usd, unit="USD"),
            cost_filters={"TagKeyValue": [f"user:{key}${value}"]}),
            notifications_with_subscribers=[alert("ACTUAL", 80), alert("FORECASTED", 100)])
