import sys
sys.path.append('src')
# print(sys.path)
# print(sys.version)


from TextSummerization.pipeline.data_ingestion_stage1 import DataIngestionTrainingPipeline
from TextSummerization.pipeline.data_validation_stage2 import DataValidationTrainingPipeline
from TextSummerization.pipeline.data_transformation_stage3 import DataTransformationTrainingPipeline
from TextSummerization.pipeline.model_trainer_stage4 import ModelTrainerTrainingPipeline
from TextSummerization.pipeline.model_evaluation_stage5 import ModelEvaluationTrainingPipeline
from TextSummerization.logging import logger


# STAGE_NAME = "Data Ingestion stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataIngestionTrainingPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e
     
     
# STAGE_NAME = "Data Validation stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_validation = DataValidationTrainingPipeline()
#    data_validation.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e
     
     
     
# STAGE_NAME = "Data Transformation stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_transformation = DataTransformationTrainingPipeline()
#    data_transformation.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e



STAGE_NAME = "Model Trainer stage"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainerTrainingPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e




STAGE_NAME = "Model Evaluation stage"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evaluation = ModelEvaluationTrainingPipeline()
   model_evaluation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e