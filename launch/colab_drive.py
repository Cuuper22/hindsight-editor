"""Mount Google Drive on a Colab session without a TTY (works on Windows).

`colab drivemount` blocks on /dev/tty while the user approves Drive access in a
browser. This wrapper runs the same CLI command but replaces that wait with a
poll of Colab's credential-propagation dry run, and writes the consent URL to
--url-file so whoever is around can open it.

Must run under the colab CLI's own interpreter (it imports colab_cli):
  <uv tools>/google-colab-cli/Scripts/python.exe launch/colab_drive.py --config X -s NAME
"""
import argparse, builtins, json, sys, time

from colab_cli import cli
from colab_cli.commands import automation as A

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("-s", "--session", required=True)
ap.add_argument("--url-file", default=None)
ap.add_argument("--wait-min", type=float, default=30)
args = ap.parse_args()

cap = {}
_get_credentials = A.get_credentials


def _parse(resp):
    try:
        return json.loads(resp.text.split("\n", 1)[-1])
    except Exception:
        return {}


def get_credentials(*a, **k):
    creds = _get_credentials(*a, **k)
    raw = creds.request

    def request(method, url, **kw):
        resp = raw(method, url, **kw)
        if "credentials-propagation" in url and method == "POST":
            cap.update(raw=raw, url=url, kw={**kw, "params": dict(kw.get("params", {}))})
            uri = _parse(resp).get("unauthorized_redirect_uri")
            if uri:
                print("DRIVE_CONSENT_URL", uri, flush=True)
                if args.url_file:
                    open(args.url_file, "w").write(uri)
        return resp

    creds.request = request
    return creds


class _ApprovalWait:
    """Stands in for /dev/tty: returns once the dry run reports success."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def readline(self):
        # The grant is bound to this request's token, and the dry run keeps reporting
        # unauthorized even after approval. The real (dryrun=false) call returns HTTP 400
        # until the link is approved and 200 after, with no side effect before -- so poll it.
        import os
        deadline = time.time() + 60 * args.wait_min
        while time.time() < deadline:
            time.sleep(10)
            kw = {**cap["kw"], "params": {**cap["kw"]["params"], "dryrun": "false"}}
            r = cap["raw"]("POST", cap["url"], **kw)
            if getattr(r, "status_code", getattr(r, "status", None)) == 200:
                print("DRIVE_CONSENT_OK", flush=True)
                return "\n"
        print("DRIVE_CONSENT_TIMEOUT", flush=True)
        os._exit(3)                                                 # sys.exit hangs inside the ws thread


_open = builtins.open
builtins.open = lambda f, *a, **k: _ApprovalWait() if f == "/dev/tty" else _open(f, *a, **k)
A.get_credentials = get_credentials
A.INTERACTIVE_AUTOMATION_TIMEOUT_SEC = 60 * (args.wait_min + 5)
cli.app(["--config", args.config, "drivemount", "-s", args.session], standalone_mode=False)
