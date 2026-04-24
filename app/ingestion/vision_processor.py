import time
from app.core.logger import logger
from app.core.exceptions import VisionProcessingError
from app.llm.gemma_client import GemmaClient

class VisionProcessor:

    def __init__(self, llm_client: GemmaClient | None = None):
        self.llm_client = llm_client or GemmaClient()
        logger.info("VisionProcessor initialized with GemmaClient")

    def process(self, parsed_result: dict) -> dict:
        images = parsed_result.get("images", [])
        if not images:
            logger.info(f"No images found in {parsed_result['arxiv_id']}")
            return parsed_result
        logger.info(f"Processing {len(images)} images for {parsed_result['arxiv_id']}")
        for i, image in enumerate(images):
            try:
                logger.info(f"Processing image {i + 1}/{len(images)} from page {image['page_number']}")
                description = self._describe_image(
                    image["image_b64"],
                    image.get("caption")
                )
                parsed_result["images"][i]["description"] = description
                logger.info(f"Image {i + 1} described successfully")
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Failed to describe image {i + 1}: {e}")
                parsed_result["images"][i]["description"] = "Figure description unavailable."
        return parsed_result

    def _describe_image(self, image_b64: str, caption: str = None) -> str:
        try:
            return self.llm_client.describe_image(image_b64, caption or "")
        except Exception as e:
            raise VisionProcessingError(f"Gemma vision API failed: {e}")
