"""Agent Extension — AI agent SDK.

Builder entry point.
"""


def setup(kernel):
    """Register AgentSDK and Result with the kernel."""
    from Agent._lib import AgentSDK, Result
    kernel.alias["AgentSDK"] = AgentSDK
    kernel.alias["Result"] = Result
