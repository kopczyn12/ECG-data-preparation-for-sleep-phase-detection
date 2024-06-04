import hydra
from omegaconf import DictConfig, OmegaConf
from utils.logger import initialize_logger, logger
from utils.model_utils import model_pipeline

@hydra.main(config_path="config", config_name="default", version_base="1.2")
def pipeline(cfg: DictConfig) -> None:
    """
    Execute the appropriate pipeline based on the configuration.

    This function initializes logging, checks the pipeline name specified in the configuration,
    and runs the corresponding pipeline function. Currently, it supports the 'model_pipeline' pipeline.

    Args:
        cfg (DictConfig): The configuration object containing settings and parameters for the pipeline.

    Raises:
        ValueError: If the pipeline name specified in the configuration is invalid.
    """
    # Initialize logger
    initialize_logger(cfg)
    if cfg.pipeline.name == "model_training":
        logger.info("Configuration:\n{}".format(OmegaConf.to_yaml(cfg)))

        # Train and evaluate the model
        logger.info("Starting pipeline")
        model_pipeline(cfg)
        logger.info("Pipeline completed.")
    else:
        logger.error("Invalid pipeline name. Exiting...")
        raise ValueError("Invalid pipeline name specified in the configuration.")

if __name__ == "__main__":
    pipeline()
