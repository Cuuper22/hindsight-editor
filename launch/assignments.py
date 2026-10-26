"""List or release this account's Colab VM assignments (needs the colab CLI's interpreter).

  python assignments.py list            -> one endpoint per line
  python assignments.py unassign EP...  -> release those VMs
"""
import sys
from colab_cli.common import state

state.config_path = None
if sys.argv[1] == "list":
    for a in state.client.list_assignments():
        print(a.endpoint)
else:
    for ep in sys.argv[2:]:
        state.client.unassign(ep)
        print("unassigned", ep)
