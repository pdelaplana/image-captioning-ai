import logging
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)

def load_model(model_name):
    try:
        logger.info(f"Loading model {model_name}")
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise e
    

def generate_captions(model, processor, image, device, text=None):
    try:
        # Prepare the inputs for captioning
        if text is not None:
            # Conditional image captioning
            inputs = processor(images=image, text=text,  return_tensors="pt").to(device)
        else:
            # Unconditional image captioning
            inputs = processor(images=image, text=text,  return_tensors="pt").to(device)

        # Generate the caption
        output = model.generate(**inputs, num_beams=15, num_return_sequences=1, do_sample=True, top_p=0.9, early_stopping=True, max_new_tokens=200)  # Generate 5 captions

        # Decode the generated captions
        captions = [processor.decode(out, skip_special_tokens=True) for out in output] 

        return captions
    except Exception as e:
        logger.exception(f"Error during caption generation: {e}")
        raise e