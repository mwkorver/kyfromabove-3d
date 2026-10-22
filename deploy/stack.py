"""The AWS stacks: the tile step's Lambda (``TileStepStack``, each tile's
inputs: product 1 and the frame list), the queue run on EC2 Spot fed by SQS
(``QueueRunStack``, refactor step 4: the measure and build jobs of
``kyfromabove_3d.cloud_jobs``), the CloudFront viewer (``ViewerStack``) and
the monthly budget (``BudgetStack``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from aws_cdk import CfnOutput, Duration, Fn, RemovalPolicy, Size, Stack
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sqs as sqs
from constructs import Construct

ROOT = Path(__file__).resolve().parents[1]

#: The EventBridge pattern for a finished piece: its marker, last to be written.

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


#: nginx for the frame cache, from ECR Public (no Docker Hub pull limits)
COG_CACHE_NGINX = "public.ecr.aws/nginx/nginx:1.27-alpine"


#: The queue run's machines: Graviton, 16 processors, 32-64 GB, in order of
#: preference -- Graviton4 first: on Spot it costs about what Graviton3 does
#: an hour and does more work -- taken in that order where a zone has Spot
#: capacity for it (capacity-optimized-prioritized); plain capacity-optimized
#: ignored the order and chose c7g (cities1, 2026-10-22).
QUEUE_INSTANCES = ("c8g.4xlarge", "m8g.4xlarge", "c7g.4xlarge", "m7g.4xlarge")
QUEUE_MAX_PRICE = "0.45"
#: A machine's hard end: however the run goes, none lives past this.
QUEUE_LIFETIME_MIN = 760
#: The worker takes no new job after this, and finishes the ones it has (each at
#: most 2 hours) before the machine's hard end above.
QUEUE_WORKER_HOURS = 10


def queue_steps(max_machines: int, slots: int, most: int = 18):
    """``(lower, upper, machines)`` for the queue's scaling: ceil(work / slots)
    machines, up to ``max_machines``; at most ``most`` steps (a step policy
    allows 20 with the zero step), larger fleets in coarser counts. The last
    step is open above (``upper`` None)."""
    n = min(max_machines, most)
    counts = sorted({max(1, round(k * max_machines / n)) for k in range(1, n + 1)})
    out, prev = [], 0
    for i, c in enumerate(counts):
        out.append((prev * slots + 1, None if i == len(counts) - 1 else c * slots + 1, c))
        prev = c
    return out


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
        # a busy machine protects itself from scale-in (queue_worker.asg_protector); the
        # group's ARN would make the role depend on the group that uses it, so any group
        role.add_to_policy(iam.PolicyStatement(
            actions=["autoscaling:DescribeAutoScalingInstances", "autoscaling:SetInstanceProtection"], resources=["*"]))
        sg = ec2.SecurityGroup(self, "NoInbound", vpc=vpc, allow_all_outbound=True,
                               description="kyfromabove-3d queue run: no inbound; out to S3, SQS, ECR and KyFromAbove")

        ud = ec2.UserData.for_linux()
        worker = (f"docker run --name worker --network host --entrypoint python -w /var/task "
                  f"-e OMP_NUM_THREADS=1 -e COG_CACHE_URL=http://127.0.0.1:8001 -e STORE=s3://{bucket_name} "
                  f"-e WORK_DIR=/var/tmp/work -e AWS_REGION={self.region} -v /var/tmp/work:/var/tmp/work "
                  f"{image.image_uri} scripts/queue_worker.py --queue-url {queue.queue_url} --run {run} "
                  f"--slots {slots} --idle-exit-s 1800 --timeout-s 7200 --max-hours {QUEUE_WORKER_HOURS} "
                  f"> /var/tmp/worker.log 2>&1")
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
            f"s3://{bucket_name}/runs/{run}/logs/$(cat /var/lib/cloud/data/instance-id).log || true",
            # the worker has stopped (the queue idle, or the Spot notice): the machine goes too
            # (shutdown terminates it), and the group starts a fresh one while jobs remain --
            # a machine left up without its worker held its place and took no jobs (trial1)
            "shutdown -h now")
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
                    spot_allocation_strategy=autoscaling.SpotAllocationStrategy.CAPACITY_OPTIMIZED_PRIORITIZED,
                    spot_max_price=QUEUE_MAX_PRICE)),
            # a deploy that changes the machines (a new image, a new script) replaces the
            # running ones: they kept the old image and took jobs it could not run (trial1:
            # 21 publish jobs parked in seconds). Their jobs in flight come back to the queue.
            update_policy=autoscaling.UpdatePolicy.rolling_update(
                max_batch_size=max_machines, min_instances_in_service=0, pause_time=Duration.seconds(0)))
        # all the machines while the queue holds work, none when it is empty; work in flight
        # counts (a job being built is not visible, and its machine must stay)
        work = cw.MathExpression(expression="visible + flight", period=Duration.minutes(1), using_metrics={
            "visible": queue.metric_approximate_number_of_messages_visible(period=Duration.minutes(1)),
            "flight": queue.metric_approximate_number_of_messages_not_visible(period=Duration.minutes(1))})
        # as many machines as the work needs -- one per ``slots`` jobs, up to all of
        # them -- so a run's tail does not keep (or relaunch) every machine for one
        # straggler; busy machines protect themselves from scale-in (queue_worker)
        asg.scale_on_metric("ByQueue", metric=work, adjustment_type=autoscaling.AdjustmentType.EXACT_CAPACITY,
                            scaling_steps=[autoscaling.ScalingInterval(upper=0, change=0)] + [
                                autoscaling.ScalingInterval(lower=lo, upper=hi, change=m)
                                for lo, hi, m in queue_steps(max_machines, slots)],
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
