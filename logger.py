

from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        #self.writer = SummaryWriter()

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        #tf.summary.scalar(tag, value, step=step)
        #self.writer.add_scalar(tag, value, step)
        #self.writer.flush()