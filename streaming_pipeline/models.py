import hashlib
from typing import List, Optional, Tuple
from abc import ABCMeta, abstractmethod
from pydantic import BaseModel
from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
)
from unstructured.staging.huggingface import chunk_by_attention_window

from streaming_pipeline.embeddings import EmbeddingModelSingleton

class Document(BaseModel, metaclass=ABCMeta):
    """
    A Pydantic abstract model representing a document.

    Attributes:
        id (str): The ID of the document.
        metadata (dict): The metadata of the document.
        text (list): The text of the document.
        chunks (list): The chunks of the document.
        embeddings (list): The embeddings of the document.

    Methods:
        to_payloads: Returns the payloads of the document.
        compute_chunks: Computes the chunks of the document.
        compute_embeddings: Computes the embeddings of the document.
    """

    doc_id: str = None
    doc_metadata: dict = {}
    doc_text: list = []
    doc_chunks: list = []
    doc_embeddings: list = []
    doc_transformed: bool = False

    @abstractmethod
    def _transform(self):
        """Abstract method to transform data and fill text and metadata attributes.
        """
        pass

    def transform(self) -> "Document":
        """
        Transform the data and fill id, text, and metadata attributes.

        Returns:
            Document: The document object with the transformed data.
        """
        self._transform()
        self.doc_transformed = True
        return self

    def set_id(self, id: str):
        """
        Sets the ID of the document.

        Args:
            id (str): The ID of the document.
        """

        self.doc_id = id
    
    def update_metadata(self, metadata: dict):
        """
        Updates the metadata of the document.

        Args:
            metadata (dict): The metadata to update.
        """

        self.doc_metadata.update(metadata)
    
    def update_text(self, text: list):
        """
        Updates the text of the document.

        Args:
            text (list): The text to update.
        """

        self.doc_text.extend(text)

    def to_payloads(self) -> Tuple[List[str], List[dict]]:
        """
        Returns the payloads of the document.

        Returns:
            Tuple[List[str], List[dict]]: A tuple containing the IDs and payloads of the document.
        """
        assert self.doc_transformed, "Document must be transformed before creating payloads."
        payloads = []
        ids = []
        for chunk in self.doc_chunks:
            payload = self.doc_metadata.copy()
            payload.update({"text": chunk})
            # Create the chunk ID using the hash of the chunk to avoid storing duplicates.
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()

            payloads.append(payload)
            ids.append(chunk_id)

        return ids, payloads

    def compute_chunks(self, model: EmbeddingModelSingleton) -> "Document":
        """
        Computes the chunks of the document.

        Args:
            model (EmbeddingModelSingleton): The embedding model to use for computing the chunks.

        Returns:
            Document: The document object with the computed chunks.
        """
        assert self.doc_transformed, "Document must be transformed before computing chunks."
        for item in self.doc_text:
            chunked_item = chunk_by_attention_window(
                item, model.tokenizer, max_input_size=model.max_input_length
            )
            print( f"item: {item[0:10]} has {len(chunked_item)} chunks" )
            self.doc_chunks.extend(chunked_item)

        return self

    def compute_embeddings(self, model: EmbeddingModelSingleton) -> "Document":
        """
        Computes the embeddings for each chunk in the document using the specified embedding model.

        Args:
            model (EmbeddingModelSingleton): The embedding model to use for computing the embeddings.

        Returns:
            Document: The document object with the computed embeddings.
        """
        assert self.doc_transformed, "Document must be transformed before computing embeddings."
        for chunk in self.doc_chunks:
            embedding = model(chunk, to_list=True)

            self.doc_embeddings.append(embedding)

        return self

    
class WikipediaArticle(Document):
    """
    Represents a wikipedia article.

    Attributes:
        id (int): Wikipedia article ID
        title (str): Title of the article
        url (Optional[str]): URL of article (if applicable)
        text (str): Text of the wikipedia article (might contain HTML)
    """
    id: int
    url: Optional[str]
    title: str
    text: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _transform(self) :
        """
        Transform the data and fill id, text, and metadata attributes.
        """

        cleaned_text = clean_non_ascii_chars(
            replace_unicode_quotes(clean(self.text))
        )
        cleaned_title = clean_non_ascii_chars(
            replace_unicode_quotes(clean(self.title))
        )

        self.set_id(id=hashlib.md5(self.text.encode()).hexdigest())
        self.update_metadata(metadata={"title": cleaned_title, "url": self.url})
        self.update_text(text=[cleaned_title, cleaned_text])




