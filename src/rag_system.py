import sys
import os
# Add the `src` directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from create_data_embeddings import create_embeddings
from embeddings_utils import search_similar_text, search_similar_image, merge_results
from gpt_utils import GPTClient

COLLECTION_NAME = "medical_images_text"

class MultimodalRAGSystem:
   collection_name: str

   def __init__(self):
       self.gpt_client = GPTClient()
       self.qdrant_client = create_embeddings(COLLECTION_NAME)
       self.collection_name = COLLECTION_NAME

   def process_query(self, query, query_image_path=None, top_k=3):
       # 1. Text-based search if query is textual
       search_results_text = search_similar_text(self.collection_name, self.qdrant_client, query, limit=top_k)

       # 2. Image-based search if a query image is provided
       search_results_image = []
       if query_image_path:  # Only perform image retrieval if an image path is provided
           search_results_image = search_similar_image(self.collection_name, self.qdrant_client, query_image_path, limit=top_k)

       # 3. Combine both results - merging text and image results
       combined_results = merge_results(search_results_text, search_results_image)

       # 4. Query GPT with the context and images
       gpt_response = self.gpt_client.query(query, combined_results, query_image_path)

       # 5. Process and return the response
       return self.gpt_client.process_response(gpt_response)