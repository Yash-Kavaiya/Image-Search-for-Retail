"""
Image Search Service
Handles similarity search for images against product database
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from .embedding_service import EmbeddingService
from .product_database import ProductDatabase, Product


class ImageSearchService:
    """Service for performing image-based product search"""
    
    def __init__(self, product_database: ProductDatabase):
        """
        Initialize image search service
        
        Args:
            product_database: Initialized product database with embeddings
        """
        self.product_database = product_database
        self.embedding_service = EmbeddingService()
    
    def search_by_image(self, image_data: bytes, mime_type: str, top_k: int = 4) -> List[Dict]:
        """
        Search for similar products using image
        
        Args:
            image_data: Image data as bytes
            mime_type: MIME type of the image
            top_k: Number of similar products to return
            
        Returns:
            List of dictionaries containing product information and similarity scores
        """
        try:
            # Validate image
            if not self.embedding_service.validate_image(image_data):
                raise ValueError("Invalid image data")
            
            # Resize image if needed to reduce API costs
            resized_image = self.embedding_service.resize_image(image_data)
            
            # Generate embedding for the input image
            query_embedding = self.embedding_service.generate_image_embedding(resized_image, mime_type)
            
            # Get product embeddings matrix
            embeddings_matrix = self.product_database.get_embeddings_matrix()
            if embeddings_matrix is None:
                raise ValueError("Product embeddings not available")
            
            # Calculate similarities
            similarities = self.calculate_similarities(query_embedding, embeddings_matrix)
            
            # Get top similar products
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            products = self.product_database.get_all_products()
            
            for idx in top_indices:
                if idx < len(products):
                    product = products[idx]
                    similarity_score = float(similarities[idx])
                    
                    result = {
                        "product_id": product.uniq_id,
                        "product_name": product.product_name,
                        "description": product.description,
                        "brand": product.brand,
                        "category": product.category,
                        "list_price": product.list_price,
                        "sale_price": product.sale_price,
                        "product_url": product.product_url,
                        "similarity_score": similarity_score
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in image search: {str(e)}")
            raise
    
    def search_by_text(self, text_query: str, top_k: int = 4) -> List[Dict]:
        """
        Search for similar products using text query
        
        Args:
            text_query: Text query for search
            top_k: Number of similar products to return
            
        Returns:
            List of dictionaries containing product information and similarity scores
        """
        try:
            # Generate embedding for the text query
            query_embeddings = self.embedding_service.generate_text_embeddings([text_query])
            query_embedding = query_embeddings[0]
            
            # Get product embeddings matrix
            embeddings_matrix = self.product_database.get_embeddings_matrix()
            if embeddings_matrix is None:
                raise ValueError("Product embeddings not available")
            
            # Calculate similarities
            similarities = self.calculate_similarities(query_embedding, embeddings_matrix)
            
            # Get top similar products
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            products = self.product_database.get_all_products()
            
            for idx in top_indices:
                if idx < len(products):
                    product = products[idx]
                    similarity_score = float(similarities[idx])
                    
                    result = {
                        "product_id": product.uniq_id,
                        "product_name": product.product_name,
                        "description": product.description,
                        "brand": product.brand,
                        "category": product.category,
                        "list_price": product.list_price,
                        "sale_price": product.sale_price,
                        "product_url": product.product_url,
                        "similarity_score": similarity_score
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in text search: {str(e)}")
            raise
    
    def search_by_multimodal(self, text_query: str, image_data: bytes, mime_type: str, top_k: int = 4) -> List[Dict]:
        """
        Search for similar products using both text and image
        
        Args:
            text_query: Text query for search
            image_data: Image data as bytes
            mime_type: MIME type of the image
            top_k: Number of similar products to return
            
        Returns:
            List of dictionaries containing product information and similarity scores
        """
        try:
            # Validate image
            if not self.embedding_service.validate_image(image_data):
                raise ValueError("Invalid image data")
            
            # Resize image if needed
            resized_image = self.embedding_service.resize_image(image_data)
            
            # Generate multimodal embedding
            query_embedding = self.embedding_service.generate_multimodal_embedding(
                text_query, resized_image, mime_type
            )
            
            # Get product embeddings matrix
            embeddings_matrix = self.product_database.get_embeddings_matrix()
            if embeddings_matrix is None:
                raise ValueError("Product embeddings not available")
            
            # Calculate similarities
            similarities = self.calculate_similarities(query_embedding, embeddings_matrix)
            
            # Get top similar products
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            products = self.product_database.get_all_products()
            
            for idx in top_indices:
                if idx < len(products):
                    product = products[idx]
                    similarity_score = float(similarities[idx])
                    
                    result = {
                        "product_id": product.uniq_id,
                        "product_name": product.product_name,
                        "description": product.description,
                        "brand": product.brand,
                        "category": product.category,
                        "list_price": product.list_price,
                        "sale_price": product.sale_price,
                        "product_url": product.product_url,
                        "similarity_score": similarity_score
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in multimodal search: {str(e)}")
            raise
    
    def calculate_similarities(self, query_embedding: np.ndarray, embeddings_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarities between query embedding and all product embeddings
        
        Args:
            query_embedding: Query embedding vector
            embeddings_matrix: Matrix of all product embeddings
            
        Returns:
            Array of similarity scores
        """
        try:
            # Reshape query embedding to 2D if needed
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, embeddings_matrix)
            
            # Return as 1D array
            return similarities.flatten()
            
        except Exception as e:
            print(f"Error calculating similarities: {str(e)}")
            raise
    
    def get_similar_products(self, product_id: str, top_k: int = 4) -> List[Dict]:
        """
        Get products similar to a given product
        
        Args:
            product_id: ID of the reference product
            top_k: Number of similar products to return
            
        Returns:
            List of dictionaries containing similar product information
        """
        try:
            # Find the product
            product = self.product_database.get_product_by_id(product_id)
            if not product or product.embedding is None:
                raise ValueError(f"Product with ID {product_id} not found or has no embedding")
            
            # Get embeddings matrix
            embeddings_matrix = self.product_database.get_embeddings_matrix()
            if embeddings_matrix is None:
                raise ValueError("Product embeddings not available")
            
            # Calculate similarities
            similarities = self.calculate_similarities(product.embedding, embeddings_matrix)
            
            # Get top similar products (excluding the product itself)
            products = self.product_database.get_all_products()
            
            # Create list of (similarity, index) pairs and sort
            similarity_pairs = [(sim, idx) for idx, sim in enumerate(similarities)]
            similarity_pairs.sort(key=lambda x: x[0], reverse=True)
            
            results = []
            for sim, idx in similarity_pairs:
                if len(results) >= top_k:
                    break
                
                if idx < len(products) and products[idx].uniq_id != product_id:
                    similar_product = products[idx]
                    result = {
                        "product_id": similar_product.uniq_id,
                        "product_name": similar_product.product_name,
                        "description": similar_product.description,
                        "brand": similar_product.brand,
                        "category": similar_product.category,
                        "list_price": similar_product.list_price,
                        "sale_price": similar_product.sale_price,
                        "product_url": similar_product.product_url,
                        "similarity_score": float(sim)
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error getting similar products: {str(e)}")
            raise
    
    def search_by_category_and_image(self, image_data: bytes, mime_type: str, category: str, top_k: int = 4) -> List[Dict]:
        """
        Search for similar products within a specific category using image
        
        Args:
            image_data: Image data as bytes
            mime_type: MIME type of the image
            category: Product category to search within
            top_k: Number of similar products to return
            
        Returns:
            List of dictionaries containing product information and similarity scores
        """
        try:
            # Get all results first
            all_results = self.search_by_image(image_data, mime_type, top_k * 3)  # Get more to filter by category
            
            # Filter by category
            category_results = [
                result for result in all_results 
                if category.lower() in result["category"].lower()
            ]
            
            # Return top k results from the category
            return category_results[:top_k]
            
        except Exception as e:
            print(f"Error in category-based image search: {str(e)}")
            raise