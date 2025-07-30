# Semantic Search with Qdrant

## ⚙️ Project Overview

Vector search enables machines to retrieve information based on meaning, not just keywords. By converting multimodal data—such as text, images, or audio—into dense vector embeddings, we can compare content semantically using metrics like cosine similarity.

This project explores the fundamentals of vector search and its practical implementation using [**Qdrant**](https://qdrant.tech/), a high-performance vector database. It focuses on real-world applications in Large Language Model (LLM) pipelines, especially Retrieval-Augmented Generation (RAG).

Unlike traditional keyword-based retrieval, **vector search allows you to:**

- Match semantically similar content, even if it uses different wording (e.g., *“bat”* vs. *“flying mouse”*).
- Perform search across multimodal data types (e.g., text, images, video, or audio).

In a typical RAG workflow—**Search → Build Prompt → LLM**—the search step is modular. [**Qdrant**](https://qdrant.tech/) seamlessly replaces keyword search with vector-based retrieval, improving the contextual relevance of prompts passed to the LLM.

#### Resources: 
- [**Qdrant Github**](https://github.com/qdrant/qdrant_demo/)
- [**FastEmbed**](https://qdrant.tech/documentation/fastembed/)
- [**Manuals**](https://qdrant.tech/articles/vector-search-manuals/)
- [**Qdrant Internals**](https://qdrant.tech/articles/qdrant-internals/)
- [**RAG & GenAI with Qdrant**](https://qdrant.tech/articles/rag-and-genai/)
- [**Article: "Built for Vector Search"**](https://qdrant.tech/articles/dedicated-vector-search/)
- [**Practical Examples**](https://qdrant.tech/articles/practicle-examples/)

## ⚙️ Qdrant
[**Qdrant**](https://qdrant.tech/) is an **open-source vector search engine** written in Rust, designed to make vector search scalable, fast, and production-ready for solutions involving millions or billions of vectors. Dedicated vector search solutions like [**Qdrant**](https://qdrant.tech/) are needed for:

- Scalability of vector search.
- Staying in sync with the latest research and best practices in vector search
- Utilizing full vector search capabilities beyond simple semantic similarity.

- To make production-level vector search at scale;
- To stay in sync with the latest trends and best practices;
- To fully use vector search capabilities (including those beyond simple similarity search).

[**Qdrant's**](https://qdrant.tech/) dashboard allows for visualizing data points. You can see uploaded payloads and vectors. More importantly, it projects the high-dimensional vectors to 2D, enabling you to visually study how points from different categories (e.g., courses) are semantically close or different based on their text content. This visual representation helps understand patterns in unstructured data and how vector search finds closest neighbors.

## ⚙️ FastEmbed: Qdrant's Library

[**FastEmbed**](https://qdrant.tech/documentation/fastembed/) is recommended for its deep integration with [**Qdrant**](https://qdrant.tech/), simplifying the process of handling vectors and format conversions. It's lightweight, CPU-friendly, and uses ONNX runtime, making it faster than some alternatives. FastEmbed also supports local inference, meaning it doesn't incur external costs beyond your machine's resources. 
**It supports various embedding types:**
- Dense text embeddings: The most classical ones used in vector search (e.g., for semantic similarity today).
- Sparse embeddings: Important for hybrid search.
- Multi-vectors and image embeddings.

## ⚙️ Workflow

The following steps outline how to implement vector search with [**Qdrant**](https://qdrant.tech/) and integrate it into a Retrieval-Augmented Generation (RAG) pipeline:

1. **Initialize the Qdrant Client**  
   Connect to your running [**Qdrant**](https://qdrant.tech/) instance using the appropriate host and API configuration.

2. **Define a Collection**  
   Create a [**Qdrant**](https://qdrant.tech/) collection with the required vector size and distance metric (e.g., cosine). Optionally, define indexes for metadata fields if you plan to apply filters during search.

3. **Embed and Index Your Data**  
   - Iterate through your dataset or document list.  
   - For each item, combine relevant fields (e.g., `question + answer`) into a single text string.  
   - Use [`fastembed`](https://github.com/qdrant/fastembed) with your chosen model (e.g., `GinaAI/embedding-model`) to generate vector embeddings.  
   - Store each vector in [**Qdrant**](https://qdrant.tech/) along with its associated payload (e.g., original text, metadata like course name or section ID).

4. **Build a Search Function**  
   Create a `vector_search()` function that takes a query string and runs a `qdrant_client.query_points()` operation:  
   - Embed the query using the same embedding model.  
   - Optionally apply filters based on metadata (e.g., search only within a course or topic).  
   - Parse and return the relevant fields from the search results (e.g., the matched document text or metadata).

5. **Plug Into Your RAG Pipeline**  
   Swap out the search component in your RAG workflow with your `vector_search()` function.  
   This modular design makes it easy to compare or switch between different retrieval methods (e.g., lexical vs. semantic).

## ⚙️ Building a Qdrant Collections

To create a [**Qdrant**](https://qdrant.tech/) collection, you’ll need to define the following:

- **Collection name** – e.g., `zoom_camp_rag`, `zoom_camp_faq`
- **Vector size** – dimensionality of the embedding vectors (e.g., `512`)
- **Distance metric** – how similarity is calculated (e.g., `cosine`)

Once the collection is initialized, you can begin embedding and indexing your data.

The `upsert` operation in [**Qdrant**](https://qdrant.tech/) handles both:

1. Embedding your text using the selected model, and  
2. Uploading the resulting vectors (along with payload metadata) to the collection.

For larger datasets, the [**Qdrant**](https://qdrant.tech/) Python client offers optimized methods such as `upload_collection()` and `upload_points()` to support batch inserts and enable parallel processing for faster indexing.


## ⚙️ Dependencies 

| **Package**              | **Function / Use Case**                              |
|--------------------------|------------------------------------------------------|
| [`tqdm`](https://pypi.org/project/tqdm/)               | Progress bars for loops                              |
| [`notebook`](https://pypi.org/project/notebook/)       | Jupyter Notebook support                             |
| [`openai`](https://platform.openai.com/docs)           | Interface with OpenAI API for LLM tasks              |
| [`minsearch`](https://github.com/minsearch)     | Lightweight lexical search library                   |
| [`pandas`](https://pypi.org/project/pandas/)           | Data manipulation and analysis                       |
| [`scikit-learn`](https://pypi.org/project/scikit-learn/)| Machine learning utilities and vector prep           |
| [`ipywidgets`](https://pypi.org/project/ipywidgets/)   | Interactive widgets for notebooks                    |
| [`ipykernel`](https://pypi.org/project/ipykernel/)     | Kernel backend for running Python in notebooks       |
| [`docker`](https://www.docker.com) | Containerization support (for local [**Qdrant**](https://qdrant.tech/), etc.)    |
| [`numpy`](https://pypi.org/project/numpy/)             | Numerical operations and array manipulation          |


#### Qdrant Setup 
To work with [**Qdrant**](https://qdrant.tech/) in Python, install these libraries in your virtual environment:

```bash
pip install -q "qdrant-client[fastembed]>=1.14.2"
```

**Import Libraries:** 

```python
from qdrant_client import QdrantClient, models
```

- **qdrant-client**: The official Python client for connecting to [**Qdrant**](https://qdrant.tech/). Official clients are also available for other languages.
- **fastembed**: [**Qdrant's**](https://qdrant.tech/) own library specifically for vectorizing data. It simplifies the process of turning data into vectors and uploading them to [**Qdrant**](https://qdrant.tech/). FastEmbed uses ONNX runtime, making it lightweight and CPU-friendly, often faster than PyTorch Sentence Transformers, and supports local inference.

## ⚙️ Getting Started  

#### 1. Clone the Repository
```bash
git clone https://github.com/milanimcgraw/Vector-Search-with-Qdrant.git
cd Vector-Search-with-Qdrant
```
#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
#### 3. Run each cell in order in a GPU-backed notebook or Codespaces environment!

## ⚙️ Qdrant Setup (Docker)

[**Qdrant**](https://qdrant.tech/) is flexible and can be run in various ways, including on your own infrastructure, Kubernetes, or in a managed cloud. For local setup, you can use a **Docker container**.

**Pull the image and start the container using the following commands**:

```bash
docker pull qdrant/qdrant

docker run -p 6333:6333 -p 6334:6334 \
   -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
   qdrant/qdrant
```

**Listening at ports:***

- 6333 – REST API port
- 6334 – gRPC API port

***Initialize Client:**

```python
client = QdrantClient("http://localhost:6333") 
```

To help you explore your data visually, [**Qdrant**](https://qdrant.tech/) provides a built-in [**Web UI**](http://localhost:6333/dashboard), available in both [**Qdrant**](https://qdrant.tech/) Cloud and local instances. You can use it to inspect collections, check system health, and even run simple queries.

---

# ⚙️ Additional Information

### Key Entities: Points and Collections

- **Point**  
  A ***point*** represents a single data item stored in [**Qdrant**](https://qdrant.tech/). In the context of a course-based Q&A system, each point might represent an answer, along with its associated metadata.

  **Each point consists of:**
  - A unique **ID** (e.g., an integer or UUID)
  - An **embedding vector** (or multiple vectors), typically generated by a model like `GinaAI`, representing the semantic meaning of the content
  - An optional **payload**, also known as metadata (e.g., course name, section ID, topic)

Collections define the vector size and distance metric used across all points within them.

- **📁 Collection**  
  A ***collection*** is a logical container that holds related points. Each collection is typically scoped to a single conceptual or business use case—for example, a course FAQ system or support chatbot knowledge base.

### Embeddings vs. Metadata (Payload)

[**Qdrant**](https://qdrant.tech/) stores two types of data for each point: **embeddings** and **metadata** (also called *payload*). Each serves a distinct purpose in the search process:

- **Embeddings**  
  These are the vectorized representations of data used for semantic search. In a Q&A system, both questions and answers are ideal candidates for embedding—allowing the system to retrieve conceptually similar content even when different words are used.

  > Example: *"flying mouse"* might return results related to *"bat"* due to shared semantic meaning.

- **Metadata (Payload)**  
  Metadata is used for structured filtering. While not involved in semantic similarity scoring, it enables users to narrow down search results based on specific attributes—such as course name, section, or topic.

  > Example: Retrieve only answers from `Section 3` of the `Biology 101` course.


### Semantic Search

Once your data is indexed in [**Qdrant**](https://qdrant.tech/), semantic search enables retrieval based on meaning rather than keywords. The process follows these steps:

1. **Receive the Query:** A user submits a question or search prompt.

2. **Embed the Query:** The input is transformed into a vector using the same embedding model used for the stored data.

3. **Compare Vectors:** [**Qdrant**](https://qdrant.tech/) compares the query vector against all stored vectors using the configured distance metric (e.g., **cosine similarity**).

4. **Rank by Similarity:** The closest matches—those with the highest semantic similarity—are identified.

5. **Return Results:** A ranked list of the most relevant answers is returned, along with their associated metadata or payloads (if desired).


### Approximate Nearest Neighbor (ANN) Search

In large-scale vector search, finding the *exact* nearest neighbor for a given query can be computationally expensive. To balance speed and accuracy, most vector databases—including [**Qdrant**](https://qdrant.tech/), use **Approximate Nearest Neighbor (ANN)** algorithms.

ANN search finds results that are *close enough* to the true nearest neighbors, significantly improving performance while maintaining high-quality results.

- ANN enables scalable, high-speed semantic search even across millions of vectors.
- The trade-off is a small loss in accuracy in favor of much faster retrieval times.
- [**Qdrant**](https://qdrant.tech/) uses a custom ANN implementation optimized for both speed and recall, making it well-suited for real-time applications like RAG systems.


### Adding Filters for Finer Results

Semantic search can be further refined using **filters** based on metadata (payload), such as course name, section, or topic. This allows you to constrain search results to specific subsets of your data.
This filtered search behavior ensures more relevant and context-specific responses, especially in Retrieval-Augmented Generation (RAG) workflows.

- [**Qdrant**](https://qdrant.tech/) supports **filtered semantic search**, enabling precise queries without sacrificing performance.
- Filters operate on the metadata fields stored alongside each embedded vector.
- To use filters effectively, [**Qdrant**](https://qdrant.tech/) must enable a specific **filterable index** on those metadata fields.
- This approach is ideal for use cases like course Q&A systems, where users may want results scoped to a particular course or section.

> Example:  
> Searching for *"late homeworks"* in the **Data Engineering** course yields a different result than the same query in the **Machine Learning** course

### Advanced Search: Hybrid Search

**Hybrid search** refers to combining more than one retrieval method—typically lexical (keyword-based) and semantic (vector-based)—to produce more relevant and robust search results.

This approach is especially useful because user queries can vary widely. Some users enter short, keyword-heavy searches, like `"qdrant filters setup"`, while others write full natural language questions, such as `"How do I add filters in Qdrant for course-specific search?"`. A purely semantic or purely keyword-based system might perform well on one type but poorly on the other.

Hybrid search serves both types of users by leveraging the strengths of each method. It improves recall for exact matches while still capturing the intent behind more conversational queries. This makes it a powerful option for user-facing applications like chatbots, course assistants, or knowledge retrieval systems.

In practice, hybrid search often combines the results or scores from both methods—for example, by blending BM25 keyword relevance with cosine similarity from vector search.


### Keyword-Based Search with Sparse Vectors (BM25)

Keyword-based search can be implemented using **sparse vectors**—vectors where most dimensions are zero. This approach focuses on exact term matching rather than semantic similarity and is ideal for use cases involving identifiers or precise keywords (e.g., `"pandas"`). Sparse vector search with BM25 is a powerful and efficient retrieval method—especially when paired with semantic models in hybrid search setups.

Key characteristics of **sparse vectors:**

- Most dimensions are zero; only non-zero dimensions are stored, improving efficiency.
- They capture the presence or absence of specific words, not meaning.
- Ideal for matching exact terms like library names, variable names, or error codes.
- Use a flexible dictionary and support new or rare words without retraining.

[**Qdrant**](https://qdrant.tech/) supports **BM25 (Best Matching 25)** for sparse vector search, a standard algorithm in traditional information retrieval:

- **TF (Term Frequency):** Rewards documents where query terms appear more often, with diminishing returns for repetition.
- **IDF (Inverse Document Frequency):** Increases the weight of rare words and lowers it for common ones. [**Qdrant**](https://qdrant.tech/) handles IDF scoring internally via collection configuration.
- **Statistical model (not neural):** No inference required—faster and more lightweight than dense models.
- **Score properties:** BM25 scores are unbounded and not normalized. They work only for ranking within a single query and can’t be used across different queries.
- **Limitations:** BM25 often underperforms on long-form or conversational queries, where understanding meaning is more important than exact matches.

### Combining Dense and Sparse Search

[**Qdrant**](https://qdrant.tech/) allows you to configure a single collection with multiple named vectors (e.g., one for dense embeddings and one for BM25 sparse vectors). This is useful for testing different models or creating a search engine that is fast in some cases and uses more sophisticated methods for others.

You can build multi-stage retrieval pipelines using [**Qdrant's**](https://qdrant.tech/) universal query API and prefetching mechanism.

- A common pipeline is to use a fast retriever (e.g., a small dense embedding model) to retrieve an initial set of documents, then rerank them using a better but slower model (e.g., a larger neural network or a cross-encoder).
- Reranking is about reordering retrieved points based on additional rules (e.g., business rules) or using a superior model.
- Fusion is a set of methods based on combining rankings from individual search methods.
  - Reciprocal Rank Fusion (RRF) is a commonly used algorithm for fusion. It calculates an intermediate score based on individual rankings (not raw scores) from different methods (e.g., dense and sparse). RRF identifies documents that are consistently ranked well by multiple methods, even if they aren't the top result from any single method. RRF is built into [**Qdrant**](https://qdrant.tech/).


## ⚙️ License
This project is released under MIT license. 

---
> ## 📌 Credits
> 📦  This project builds on concepts and starter code introduced in the [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp/tree/main) course by [DataTalksClub](https://github.com/DataTalksClub). > > **Additional sources include:** 
> * [Cohort 2025| Vector Search using Qdrant study guide & FAQ by Nitin Gupta](https://github.com/niting9881/llm-zoomcamp/blob/main/02-vector-search/README.md)
> * [Cohort 2025| Cognee and dlt workshop study guide & FAQ by Nitin Gupta](https://github.com/niting9881/llm-zoomcamp/tree/main/02-vector-search/workshop/dlt)
> While the original instructional materials provided foundational examples, this implementation has been customized and extended.
