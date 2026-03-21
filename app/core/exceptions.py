class ArxivLensException(Exception):
    pass

class ArxivFetchError(ArxivLensException):
    pass

class PDFParseError(ArxivLensException):
    pass

class VisionProcessingError(ArxivLensException):
    pass

class ChunkingError(ArxivLensException):
    pass

class EmbeddingError(ArxivLensException):
    pass

class QdrantConnectionError(ArxivLensException):
    pass

class RetrievalError(ArxivLensException):
    pass

class GenerationError(ArxivLensException):
    pass

class PipelineError(ArxivLensException):
    pass