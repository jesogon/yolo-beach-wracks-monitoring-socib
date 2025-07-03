"""Endpoint functions to integrate your model with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.ai4os.eu/
[2]: https://github.com/ai4os-hub/ai4os-demo-app
"""

import os
import logging
import datetime
import tempfile
import shutil
import argparse
import json
import torch


from ultralytics import YOLO, settings

from aiohttp.web import HTTPException
from deepaas.model.v2.wrapper import UploadedFile

import beach_wracks_monitoring as aimodel
from . import config, responses, schemas, utils
from beach_wracks_monitoring.utils import (
    mlflow_fetch,
    mlflow_logging,
)


logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


# global var

MLFLOW_MODEL_NAME = "yolo11_beach_wracks_identification"


def get_metadata():
    """Returns a dictionary containing metadata information about the module.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.MODEL_NAME)
        print(config.BASE_PATH)
        metadata = {
            "author": config.MODEL_METADATA.get("authors"),
            "author-email": config.MODEL_METADATA.get(
                "author-emails"
            ),
            "description": config.MODEL_METADATA.get("summary"),
            "license": config.MODEL_METADATA.get("license"),
            "version": config.MODEL_METADATA.get("version"),
            "models_local": utils.ls_dirs(config.MODELS_PATH),
            # "models_remote": utils.ls_remote(), 
            "datasets": utils.generate_directory_tree(
                config.DATA_PATH
            ),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        raise HTTPException(reason=err) from err


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict(**args):
    """Performs model prediction from given input data and parameters.

    Arguments:
            **args -- Arbitrary keyword arguments from PredArgsSchema.

    Raises:
            HTTPException: Unexpected errors aim to return 50X

    Returns:
            The predicted model values json, png, pdf or mp4 file.
    """

    logger.debug("Predict with args: %s", args)
    try:
        if args["model"] is None:
            args["model"] = config.DEFAULT_MODEL_PATH  # Only segmentation is enabled

        else:
            path = os.path.join(args["model"], "weights/best.pt")
            args["model"] = utils.validate_and_modify_path(
                path, config.MODELS_PATH
            )

        task_type = args["task_type"]

        with tempfile.TemporaryDirectory() as tmpdir:
            for f in [args["files"]]:
                shutil.copy(
                    f.filename, tmpdir + "/" + os.path.basename(f.original_filename)
                )

            args["files"] = [
                os.path.join(tmpdir, t) for t in os.listdir(tmpdir)
            ]
            result = aimodel.predict(**args)
            logger.debug("Predict result: %s", result)
            logger.info(
                "Returning content_type for: %s", args["accept"]
            )
            return responses.response_parsers[args["accept"]](
                result, **args
            )

    except Exception as err:
        raise HTTPException(reason=err) from err


@utils.train_arguments(schema=schemas.TrainArgsSchema)
def train(**args):
    """
    Trains a yolo11 model using the specified arguments.

    Args:
        **args (dict): A dictionary of arguments for training the model
        defined in the schema.

    Returns:
        dict: A dictionary containing a success message and the path
        where the trained model was saved.

    Raises:
        HTTPException: If an error occurs during training.
    Note:
        - The `project` argument should correspond to the name of
        your project and should only include the project directory,
        not the full path.
        - The `name` argument specifies the subdirectory where the
        model will be saved within the project directory.
        - The `weights` argument can be used to load pre-trained
        weights from a file.
    """
    try:
        logger.info("Training model...")
        logger.debug("Train with args: %s", args)
        Enable_MLFLOW = args["Enable_MLFLOW"]
        settings.update(
            {
                "mlflow": False,
                #"datasets_dir": config.DATA_PATH,
              #  "model_dir": config.MODELS_PATH,
                "wandb": args["disable_wandb"],
            }
        )
        # Modify the model name based on task type
        args["model"] = utils.modify_model_name(
            args["model"], args["task_type"]
        )
        # Check and update data path if necessary
        base_path = os.path.join(config.DATA_PATH, "processed")
        args["data"] = utils.validate_and_modify_path(
            args["data"], base_path
        )
        task_type = args["task_type"]
        if task_type in ["det", "seg", "obb"]:
            # Check and update data paths of val and training in config.yaml
            if not utils.check_paths_in_yaml(args["data"], base_path):
                raise ValueError(
                    "The path to the either train or validation "
                    "data does not exist. Please provide a valid path."
                )

        # The project should correspond to the name of the project
        # and should only include the project directory.
        args["project"] = config.MODELS_PATH

        # The directory where the model will be saved after training
        # by joining the values of args["project"] and args["name"].
        args["name"] = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        # Check if there are weights to load from an already trained model
        # Otherwise, load the pretrained model from the model registry

        if args["weights"] is not None:
            path = utils.validate_and_modify_path(
                args["weights"], config.MODELS_PATH
            )

            model = YOLO(path)

        else:
            model = YOLO(args["model"])
        if "auto_augment" not in args:
            args["auto_augment"] = None

        device = args.get("device", "cpu")
        if device != "cpu" and not torch.cuda.is_available():
            print("CUDA is not available, switching to CPU mode.")
            device = "cpu"
        os.environ["WANDB_DISABLED"] = str(args["disable_wandb"])

        utils.pop_keys_from_dict(
            args,
            [
                "task_type",
                "disable_wandb",
                "weights",
                "device",
                "Enable_MLFLOW",
            ],
        )
        if Enable_MLFLOW:
            num_epochs = args["epochs"]
            model.train(exist_ok=True, device=device, **args)

            # Call the mlflow_logging function for MLflow-related operations
            return mlflow_logging(model, num_epochs, args)
        else:
            model.train(exist_ok=True, device=device, **args)
            return {
                f'The model was trained successfully and was saved to: \
                {os.path.join(args["project"], args["name"])}'
            }

    except Exception as err:
        logger.critical(err, exc_info=True)
        raise HTTPException(reason=err) from err


def main():
    """
    Runs above-described methods from CLI
    uses: python3 path/to/api/__init__.py method --arg1 ARG1_VALUE
     --arg2 ARG2_VALUE
    """
    method_dispatch = {
        "get_metadata": get_metadata,
        "predict": predict,
        "train": train,
    }

    chosen_method = args.method
    logger.debug("Calling method: %s", chosen_method)
    if chosen_method in method_dispatch:
        method_function = method_dispatch[chosen_method]

        if chosen_method == "get_metadata":
            results = method_function()
        else:
            logger.debug("Calling method with args: %s", args)
            del vars(args)["method"]
            if hasattr(args, "files"):
                file_extension = os.path.splitext(args.files)[1]
                args.files = UploadedFile(
                    "files",
                    args.files,
                    "application/octet-stream",
                    f"files{file_extension}",
                )
            results = method_function(**vars(args))
        print(json.dumps(results))
        logger.debug("Results: %s", results)
        return results
    else:
        print("Invalid method specified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model parameters", add_help=False
    )
    cmd_parser = argparse.ArgumentParser()
    subparsers = cmd_parser.add_subparsers(
        help='methods. Use "api.py method --help" to get more info',
        dest="method",
    )
    get_metadata_parser = subparsers.add_parser(
        "get_metadata", help="get_metadata method", parents=[parser]
    )

    predict_parser = subparsers.add_parser(
        "predict", help="commands for prediction", parents=[parser]
    )

    utils.add_arguments_from_schema(
        schemas.PredArgsSchema(), predict_parser
    )

    train_parser = subparsers.add_parser(
        "train", help="commands for training", parents=[parser]
    )

    utils.add_arguments_from_schema(
        schemas.TrainArgsSchema(), train_parser
    )

    args = cmd_parser.parse_args()

    main()

    """
    python3 api/__init__.py  train --model yolov8n.yaml\
    --task_type  det\
    --data /srv/football-players-detection-7/data.yaml\
    --Enable_MLFLOW --epochs 50
    python3 api/__init__.py  predict --files \
    /srv/yolov8_api/tests/data/det/test/cat1.jpg\
    --task_type  det --accept application/json
    """
