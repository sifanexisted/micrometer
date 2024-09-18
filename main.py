from absl import app
from absl import flags
from ml_collections import config_flags

import train
import finetune
import eval
import eval_homo
import eval_multiscale


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "configs/base.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if FLAGS.config.mode == "train":
        train.train_and_evaluate(FLAGS.config)

    elif FLAGS.config.mode.startswith("finetune"):
        finetune.train_and_evaluate(FLAGS.config)

    elif FLAGS.config.mode == "eval":
        eval.evaluate(FLAGS.config)

    # WARNING: number of GPUs should be divided by 1250
    elif FLAGS.config.mode == "eval_homo":
        eval_homo.evaluate(FLAGS.config)

    elif FLAGS.config.mode == "eval_multiscale":
        eval_multiscale.evaluate(FLAGS.config)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
