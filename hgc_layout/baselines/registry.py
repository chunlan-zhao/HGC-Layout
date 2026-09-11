"""Construction of the compared-method set."""
from typing import Dict, List

from hgc_layout.baselines.graphrnn import GraphRNNLayout
from hgc_layout.baselines.fw_fd import FWFD
from hgc_layout.baselines.layoutgpt import LayoutGPT
from hgc_layout.baselines.urbangan import UrbanGAN
from hgc_layout.planning.planner import TemplatePlanner

BASELINE_NAMES = ("LayoutGPT", "FW-FD", "UrbanGAN", "GraphRNN-Layout")


def build_baselines(planner: TemplatePlanner, names=BASELINE_NAMES) -> Dict[str, object]:
    """Instantiate the baselines by name, all sharing one planner."""
    factory = {"LayoutGPT": LayoutGPT, "FW-FD": FWFD,
               "UrbanGAN": UrbanGAN, "GraphRNN-Layout": GraphRNNLayout}
    return {n: factory[n](planner) for n in names}
