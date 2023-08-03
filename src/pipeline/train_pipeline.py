from components.data_transformation import DataTransformation, DataTransformationConfig
from components.model_trainer import ModelTrainer, ModelTrainerConfig
from components.data_ingestion import DataIngestion, DataIngestionConfig



class TrainingPipeline:
    def __init__(self):
        pass
    def initiate_training_pipeline(self):
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)
        model_trainer = ModelTrainer()
        score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(score)