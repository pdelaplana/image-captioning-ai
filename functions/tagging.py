import logging
import spacy

logger = logging.getLogger(__name__)

def tag_captions(captions):
    try:
        # Load spaCy model for POS tagging
        nlp = spacy.load("en_core_web_sm")

        # Emoji mapping based on parts of speech
        emoji_mapping = {
            "NOUN": {
                "dog": "🐕", "cat": "🐈", "tree": "🌳", "car": "🚗", "flower": "🌸",
                "sun": "🌞", "mountain": "🏔️", "beach": "🏖️", "food": "🍔"
            },
            "VERB": {
                "run": "🏃", "eat": "🍽️", "play": "🎮", "jump": "🤸", "swim": "🏊"
            },
            "ADJ": {
                "happy": "😊", "beautiful": "😍", "sad": "😢"
            }
        }

        # Process each caption with spaCy and add emojis
        captions_with_emojis = []
        for caption in captions:
            # Use spaCy to perform POS tagging on the generated caption
            doc = nlp(caption)

            # Add emojis to the caption based on POS tags and mapping
            caption_with_emojis = ""
            for token in doc:
                word = token.text
                pos = token.pos_

                # Check if the word has an emoji mapping based on its POS tag
                if pos in emoji_mapping and word.lower() in emoji_mapping[pos]:
                    emoji = emoji_mapping[pos][word.lower()]
                    caption_with_emojis += f"{word} {emoji} "
                else:
                    caption_with_emojis += f"{word} "

            # Append processed caption with emojis
            captions_with_emojis.append(caption_with_emojis.strip())


        # Return the generated captions as a JSON response
        return captions_with_emojis


    except Exception as e:
        logger.exception(f"Error during POS tagging : {e}")
        raise e