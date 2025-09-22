"""
Embedding Service Module
Handles text and image embedding generation using Google Gemini
"""
import os
import base64
import numpy as np
from typing import List, Union, Optional
from google import genai
from google.genai import types
from PIL import Image
import io


class EmbeddingService:
    """Service for generating embeddings using Google Gemini"""
    
    def __init__(self):
        """Initialize the embedding service with Gemini client"""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment variables. "
                "Get your API key from: https://aistudio.google.com/"
            )
        self.client = genai.Client(api_key=api_key)
        self.text_model = "gemini-embedding-001"
        self.multimodal_model = "gemini-multimodal-embedding-001"
    
    def generate_text_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of text strings
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of numpy arrays containing embeddings
        """
        try:
            result = self.client.models.embed_content(
                model=self.text_model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY"
                )
            )
            
            embeddings = [np.array(e.values) for e in result.embeddings]
            return embeddings
            
        except Exception as e:
            print(f"Error generating text embeddings: {str(e)}")
            raise
    
    def generate_image_embedding(self, image_data: bytes, mime_type: str) -> np.ndarray:
        """
        Generate embedding for a single image
        
        Args:
            image_data: Image data as bytes
            mime_type: MIME type of the image (e.g., 'image/jpeg')
            
        Returns:
            Numpy array containing the image embedding
        """
        try:
            # Convert image data to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create image part for Gemini
            image_part = types.Part.from_bytes(
                mime_type=mime_type,
                data=base64.b64decode(base64_image)
            )
            
            # Generate embedding
            result = self.client.models.embed_content(
                model=self.multimodal_model,
                contents=[image_part],
                config=types.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY"
                )
            )
            
            if result.embeddings:
                return np.array(result.embeddings[0].values)
            else:
                raise ValueError("No embedding generated for the image")
                
        except Exception as e:
            print(f"Error generating image embedding: {str(e)}")
            raise
    
    def generate_multimodal_embedding(self, text: str, image_data: bytes, mime_type: str) -> np.ndarray:
        """
        Generate embedding for combined text and image
        
        Args:
            text: Text description
            image_data: Image data as bytes
            mime_type: MIME type of the image
            
        Returns:
            Numpy array containing the multimodal embedding
        """
        try:
            # Convert image data to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create parts for text and image
            text_part = types.Part(text=text)
            image_part = types.Part.from_bytes(
                mime_type=mime_type,
                data=base64.b64decode(base64_image)
            )
            
            # Generate embedding
            result = self.client.models.embed_content(
                model=self.multimodal_model,
                contents=[text_part, image_part],
                config=types.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY"
                )
            )
            
            if result.embeddings:
                return np.array(result.embeddings[0].values)
            else:
                raise ValueError("No embedding generated for the multimodal content")
                
        except Exception as e:
            print(f"Error generating multimodal embedding: {str(e)}")
            raise
    
    def validate_image(self, image_data: bytes) -> bool:
        """
        Validate if the image data is a valid image
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            True if valid image, False otherwise
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            image.verify()
            return True
        except Exception:
            return False
    
    def resize_image(self, image_data: bytes, max_size: tuple = (800, 800)) -> bytes:
        """
        Resize image if it's too large
        
        Args:
            image_data: Original image data
            max_size: Maximum dimensions (width, height)
            
        Returns:
            Resized image data as bytes
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Check if resize is needed
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save resized image
                output = io.BytesIO()
                image_format = image.format or 'JPEG'
                image.save(output, format=image_format, quality=85)
                return output.getvalue()
            
            return image_data
            
        except Exception as e:
            print(f"Error resizing image: {str(e)}")
            return image_data