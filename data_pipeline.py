import hydra
from omegaconf import DictConfig, OmegaConf
from utils.logger import initialize_logger, logger
from utils.data_utils import dataset_preparation

@hydra.main(config_path="config", config_name="default", version_base="1.2")
def pipeline(cfg: DictConfig) -> None:
    """
    Execute the appropriate pipeline based on the configuration.

    This function initializes logging, checks the pipeline name specified in the configuration,
    and runs the corresponding pipeline function. Currently, it supports the 'ecg_data_preparation' pipeline.

    Args:
        cfg (DictConfig): The configuration object containing settings and parameters for the pipeline.

    Raises:
        ValueError: If the pipeline name specified in the configuration is invalid.
    """
    # Initialize logger
    initialize_logger(cfg)
    if cfg.pipeline.name == "ecg_data_preparation":
        logger.info("Starting pipeline...")
        logger.info("Configuration:\n{}".format(OmegaConf.to_yaml(cfg)))

        # Prepare dataset
        logger.info("Preparing dataset...")
        dataset_preparation(cfg)
        logger.info("Dataset preparation completed.")
        logger.info("Pipeline completed.")
    else:
        logger.error("Invalid pipeline name. Exiting...")
        raise ValueError("Invalid pipeline name specified in the configuration.")

if __name__ == "__main__":
    pipeline()