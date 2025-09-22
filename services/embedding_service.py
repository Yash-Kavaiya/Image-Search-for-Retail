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
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingService:
    """Service for generating embeddings using Google Gemini"""
    
    def __init__(self):
        """Initialize the embedding service with Gemini client"""
        api_key = os.environ.get("GEMINI_API_KEY")
        # Decide whether to use remote Gemini or local fallback.
        # By default use local fallback for reliability unless USE_REMOTE=1 and API key is present.
        prefer_remote = os.environ.get('USE_REMOTE', '0') == '1'
        if not api_key or not prefer_remote:
            # Use local fallback when no API key or remote not explicitly enabled
            self.client = None
            self.use_local_fallback = True
            # Local vectorizer for text embeddings (will be fit on corpus when needed)
            self.vectorizer: Optional[TfidfVectorizer] = None
            self.text_dim = 512
            self.multimodal_dim = 536  # text_dim + image histogram dim (24)
        else:
            self.client = genai.Client(api_key=api_key)
            self.text_model = "gemini-embedding-001"
            self.multimodal_model = "gemini-multimodal-embedding-001"
            self.use_local_fallback = False
            self.vectorizer = None
    
    def generate_text_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of text strings
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of numpy arrays containing embeddings
        """
        try:

            # If configured to use local fallback, or if remote call fails, use TF-IDF
            if self.use_local_fallback:
                use_remote = False
            else:
                use_remote = True

            if use_remote:
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
                except Exception as e_remote:
                    print(f"Remote text embedding failed ({e_remote}), falling back to local TF-IDF")

            # Local TF-IDF fallback
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(max_features=self.text_dim)
                X = self.vectorizer.fit_transform(texts)
            else:
                X = self.vectorizer.transform(texts)

            embeddings = [np.asarray(row.todense()).ravel() for row in X]
            # If TF-IDF produced lower-dim vectors, pad to text_dim
            for i, emb in enumerate(embeddings):
                if emb.size < self.text_dim:
                    pad = np.zeros(self.text_dim - emb.size, dtype=float)
                    embeddings[i] = np.concatenate([emb, pad])
            return [np.array(e) for e in embeddings]

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

            # Try remote embedding if possible, otherwise fallback to histogram
            use_remote = not self.use_local_fallback
            if use_remote:
                try:
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    image_part = types.Part.from_bytes(
                        mime_type=mime_type,
                        data=base64.b64decode(base64_image)
                    )
                    result = self.client.models.embed_content(
                        model=self.multimodal_model,
                        contents=[image_part],
                        config=types.EmbedContentConfig(
                            task_type="SEMANTIC_SIMILARITY"
                        )
                    )
                    if result.embeddings:
                        return np.array(result.embeddings[0].values)
                except Exception as e_remote:
                    print(f"Remote image embedding failed ({e_remote}), falling back to local histogram")

            # Local histogram fallback (3 channels x 8 bins = 24 dims)
            try:
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                image = image.resize((128, 128))
                arr = np.array(image)
                hist = []
                bins = 8
                for c in range(3):
                    channel = arr[:, :, c]
                    h, _ = np.histogram(channel, bins=bins, range=(0, 255), density=True)
                    hist.extend(h.tolist())
                emb = np.array(hist, dtype=float)
                # Normalize
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                return emb
            except Exception:
                # Fallback to random deterministic vector based on data hash
                h = base64.b64encode(image_data)[:32]
                rng = np.random.default_rng(int.from_bytes(h, 'little', signed=False))
                emb = rng.standard_normal(24)
                emb = emb / (np.linalg.norm(emb) + 1e-12)
                return emb

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

            # Try remote multimodal embedding first when available
            use_remote = not self.use_local_fallback
            if use_remote:
                try:
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    text_part = types.Part(text=text)
                    image_part = types.Part.from_bytes(
                        mime_type=mime_type,
                        data=base64.b64decode(base64_image)
                    )
                    result = self.client.models.embed_content(
                        model=self.multimodal_model,
                        contents=[text_part, image_part],
                        config=types.EmbedContentConfig(
                            task_type="SEMANTIC_SIMILARITY"
                        )
                    )
                    if result.embeddings:
                        return np.array(result.embeddings[0].values)
                except Exception as e_remote:
                    print(f"Remote multimodal embedding failed ({e_remote}), falling back to local multimodal")

            # Local multimodal: text TF-IDF (or existing vectorizer) + image histogram
            text_emb = self.generate_text_embeddings([text])[0]
            img_emb = self.generate_image_embedding(image_data, mime_type)
            combined = np.concatenate([text_emb[:self.text_dim], img_emb])
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
            return combined

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