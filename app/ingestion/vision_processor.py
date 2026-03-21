import time
from groq import Groq
from app.core.logger import logger
from app.core.exceptions import VisionProcessingError
from app.core.settings import settings


VISION_PROMPT = """You are analyzing a figure from an academic research paper.
{caption}
Describe this figure in detail including:
- What type of figure it is (chart, diagram, graph, table, architecture, equation)
- What data or information it shows
- Key values, trends, or findings visible
- Any labels, axes, or legends present
Be precise and technical. This description will be used for academic research queries."""


class VisionProcessor:

    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.vision_model
        logger.info(f"VisionProcessor initialized with model: {self.model}")

    def process(self, parsed_result: dict) -> dict:
        images = parsed_result.get("images", [])
        if not images:
            logger.info(f"No images found in {parsed_result['paper_id']}")
            return parsed_result
        logger.info(f"Processing {len(images)} images for {parsed_result['paper_id']}")
        for i, image in enumerate(images):
            try:
                logger.info(f"Processing image {i + 1}/{len(images)} from page {image['page_number']}")
                description = self._describe_image(
                    image["image_b64"],
                    image["ext"],
                    image.get("caption")
                )
                parsed_result["images"][i]["description"] = description
                logger.info(f"Image {i + 1} described successfully")
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Failed to describe image {i + 1}: {e}")
                parsed_result["images"][i]["description"] = "Figure description unavailable."
        return parsed_result

    def _describe_image(self, image_b64: str, ext: str, caption: str = None) -> str:
        mime_type = f"image/{ext}" if ext != "jpg" else "image/jpeg"
        caption_text = f"Caption: {caption}" if caption else "No caption available."
        prompt = VISION_PROMPT.format(caption=caption_text)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise VisionProcessingError(f"Groq vision API failed: {e}")