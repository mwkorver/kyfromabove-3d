# Security

This is a public prototype, not a maintained project, on the same footing as
[CONTRIBUTING.md](CONTRIBUTING.md). There is no disclosure process and no
response commitment. Fork it and fix it.

Three things are worth knowing before you deploy your own copy, because they
cost real money on your account:

- **The Spot queue runner spends your money.** `QueueRunStack` (`deploy/stack.py`)
  and `scripts/queue_run.py` scale EC2 Spot machines with a per-hour price
  ceiling and a hard machine lifetime (no new jobs after 10 hours, terminated at
  12 hours 40 minutes), but there is no automatic spending cap. Deploy the
  budget stack as well (`docs/CLOUD.md`, "Watching the spend"); it emails,
  it does not stop anything.
- **A deployed CloudFront viewer is public** unless you switch the password on
  (`-c viewer_password=on`). Every request to a public viewer is billed to you.
- **The viewer password is a demo gate, not authentication.** It is HTTP basic
  auth in a CloudFront Function, with one shared password, meant to keep a demo
  out of casual view and nothing more.

The default bucket name `kyfromabove-3d` in the code and docs is the author's.
Bucket names are global, so pass your own via `-c bucket=<your-bucket>` (or
`KFA_BUCKET=<your-bucket>` / `--store s3://<your-bucket>`).
