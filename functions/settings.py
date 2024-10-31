from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    blip_model_name: str = "Salesforce/blip-image-captioning-large"
    #blip_model_name: str = "Salesforce/blip-image-captioning-base"

settings = Settings()

