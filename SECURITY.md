# Security

This is a public prototype, not a maintained project, on the same footing as
[CONTRIBUTING.md](CONTRIBUTING.md). There is no disclosure process and no
response commitment. Fork it and fix it.

Three things are worth knowing before you deploy your own copy, because they
cost real money on your account:

- **The Spot build runner spends your money.** `deploy/run_spot_build.py`
  starts EC2 Spot machines and stops them before a spending cap you give it,
  but the cap is a check the script makes, not an AWS budget. Set an AWS
  Budget alarm as well.
- **A deployed CloudFront viewer is public** unless you switch the password on
  (`-c viewer_password=on`). Every request to a public viewer is billed to you.
- **The viewer password is a demo gate, not authentication.** It is HTTP basic
  auth in a CloudFront Function, with one shared password, meant to keep a demo
  out of casual view and nothing more.

The bucket name `kyfromabove-3d` in the code and docs is the author's. Bucket
names are global, so pick your own.
