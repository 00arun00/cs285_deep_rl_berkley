"""Launch sweep trials with only warnings and errors from the agent logger.

Run from ``hw1/`` with a sweep ID and optional ``--count``. The parent agent
launches the training command registered in the sweep; filtering its messages
does not change W&B output in the training subprocess.

The filter targets W&B's internal ``wandb.wandb_agent`` logger, whose integration
must be reviewed when upgrading W&B. It suppresses routine status messages,
but warnings and errors can still interrupt a live progress bar.
"""

import argparse
import logging

import wandb


class AgentWarningsOnly(logging.Filter):
    """Keep agent diagnostics without interrupting tqdm for routine status."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Allow warnings and more severe agent messages.

        Args:
            record: Agent log record being considered for emission.

        Returns:
            Whether the record has severity WARNING or above.
        """
        return record.levelno >= logging.WARNING


def main() -> None:
    """Parse launcher arguments and run the W&B sweep agent.

    Read a positional ``sweep_id`` and optional integer ``--count`` from command
    line arguments. The count limits trials for this invocation; omission leaves
    that limit unset. Launching the agent can start real training and W&B uploads.

    Install the severity filter only on the parent agent logger and remove it
    when the agent returns or raises. Training subprocess output is unaffected.
    W&B authentication, sweep lookup, and execution exceptions propagate.

    Raises:
        SystemExit: Argument parsing exits for help or invalid CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_id", help="W&B ENTITY/PROJECT/SWEEP_ID")
    parser.add_argument("--count", type=int, help="Maximum number of trials to run")
    args = parser.parse_args()

    # The agent resets its logger level, so use a filter instead of setLevel().
    # This logger belongs to the parent agent, not the training subprocess.
    logger = logging.getLogger("wandb.wandb_agent")
    severity_filter = AgentWarningsOnly()
    logger.addFilter(severity_filter)
    try:
        wandb.agent(args.sweep_id, count=args.count)
    finally:
        logger.removeFilter(severity_filter)


if __name__ == "__main__":
    main()
