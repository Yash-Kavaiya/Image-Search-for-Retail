"""
Product Database Service
Handles loading products from CSV and managing embeddings
"""
import pandas as pd
import numpy as np
import pickle
import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .embedding_service import EmbeddingService


@dataclass
class Product:
    """Product data structure"""
    uniq_id: str
    product_url: str
    product_name: str
    description: str
    list_price: float
    sale_price: float
    brand: str
    category: str
    embedding: Optional[np.ndarray] = None


class ProductDatabase:
    """Service for managing product data and embeddings"""
    
    def __init__(self, csv_path: str = "retail copy.csv", embeddings_cache_path: str = "data/embeddings_cache.pkl"):
        """
        Initialize product database
        
        Args:
            csv_path: Path to the retail CSV file
            embeddings_cache_path: Path to cache embeddings
        """
        self.csv_path = csv_path
        self.embeddings_cache_path = embeddings_cache_path
        self.embedding_service = EmbeddingService()
        self.products: List[Product] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(embeddings_cache_path), exist_ok=True)
        
    def load_products_from_csv(self) -> None:
        """Load products from CSV file"""
        try:
            df = pd.read_csv(self.csv_path)
            
            self.products = []
            for _, row in df.iterrows():
                # Clean and prepare data
                description = str(row.get('Description', '')).strip()
                product_name = str(row.get('Product_Name', '')).strip()
                category = str(row.get('Category', '')).strip()
                
                # Combine relevant text fields for embedding
                combined_text = f"{product_name}. {description}. Category: {category}"
                
                product = Product(
                    uniq_id=str(row.get('Uniq_Id', '')),
                    product_url=str(row.get('Product_Url', '')),
                    product_name=product_name,
                    description=description,
                    list_price=float(row.get('List_Price', 0)),
                    sale_price=float(row.get('Sale_Price', 0)),
                    brand=str(row.get('Brand', '')),
                    category=category
                )
                
                self.products.append(product)
                
            print(f"Loaded {len(self.products)} products from CSV")
            
        except Exception as e:
            print(f"Error loading products from CSV: {str(e)}")
            raise
    
    def generate_embeddings(self, force_regenerate: bool = False) -> None:
        """
        Generate embeddings for all products
        
        Args:
            force_regenerate: If True, regenerate even if cache exists
        """
        # Check if cached embeddings exist
        if not force_regenerate and os.path.exists(self.embeddings_cache_path):
            print("Loading cached embeddings...")
            try:
                with open(self.embeddings_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.embeddings_matrix = cached_data['embeddings_matrix']
                    
                    # Assign embeddings to products
                    for i, embedding in enumerate(cached_data['embeddings']):
                        if i < len(self.products):
                            self.products[i].embedding = embedding
                    
                    print(f"Loaded {len(cached_data['embeddings'])} cached embeddings")
                    return
            except Exception as e:
                print(f"Error loading cached embeddings: {str(e)}")
                print("Regenerating embeddings...")
        
        # Generate new embeddings
        print("Generating embeddings for all products...")
        
        # Prepare text for embedding
        texts = []
        for product in self.products:
            # Combine product information for better embedding
            combined_text = f"{product.product_name}. {product.description}. Brand: {product.brand}. Category: {product.category}"
            texts.append(combined_text)
        
        try:
            # Generate embeddings in batches to avoid API limits
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedding_service.generate_text_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
                print(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            # Assign embeddings to products
            for i, embedding in enumerate(all_embeddings):
                if i < len(self.products):
                    self.products[i].embedding = embedding
            
            # Create embeddings matrix
            self.embeddings_matrix = np.array(all_embeddings)
            
            # Cache embeddings
            cache_data = {
                'embeddings': all_embeddings,
                'embeddings_matrix': self.embeddings_matrix,
                'product_ids': [p.uniq_id for p in self.products]
            }
            
            with open(self.embeddings_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"Generated and cached {len(all_embeddings)} embeddings")
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_product_by_id(self, product_id: str) -> Optional[Product]:
        """Get product by unique ID"""
        for product in self.products:
            if product.uniq_id == product_id:
                return product
        return None
    
    def get_products_by_category(self, category: str) -> List[Product]:
        """Get products by category"""
        return [p for p in self.products if category.lower() in p.category.lower()]
    
    def search_products_by_text(self, query: str, limit: int = 10) -> List[Product]:
        """Search products by text query"""
        query_lower = query.lower()
        results = []
        
        for product in self.products:
            if (query_lower in product.product_name.lower() or 
                query_lower in product.description.lower() or 
                query_lower in product.brand.lower() or 
                query_lower in product.category.lower()):
                results.append(product)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_all_products(self) -> List[Product]:
        """Get all products"""
        return self.products
    
    def get_embeddings_matrix(self) -> Optional[np.ndarray]:
        """Get the embeddings matrix"""
        return self.embeddings_matrix
    
    def initialize(self, force_regenerate_embeddings: bool = False) -> None:
        """
        Initialize the database by loading products and generating embeddings
        
        Args:
            force_regenerate_embeddings: If True, regenerate embeddings even if cache exists
        """
        print("Initializing product database...")
        self.load_products_from_csv()
        self.generate_embeddings(force_regenerate=force_regenerate_embeddings)
        print("Product database initialized successfully!")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        return {
            "total_products": len(self.products),
            "products_with_embeddings": sum(1 for p in self.products if p.embedding is not None),
            "unique_brands": len(set(p.brand for p in self.products if p.brand)),
            "unique_categories": len(set(p.category for p in self.products if p.category))
        }