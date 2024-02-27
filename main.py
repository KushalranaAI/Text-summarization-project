import sys
sys.path.append('src')

from TextSummerization.pipeline.data_ingestion_stage1 import DataIngestionTrainingPipeline
# from TextSummerization.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
# from TextSummerization.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
# from TextSummerization.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
# from TextSummerization.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from TextSummerization.logging import logger


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e