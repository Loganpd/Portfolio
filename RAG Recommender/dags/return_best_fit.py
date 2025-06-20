from airflow.sdk import dag, task, Asset

COLLECTION_NAME = "movies"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"


@dag(schedule=[Asset("movie_vector_data")],
     params={"query_string": "a mind-bending movie"})
def return_best_fit():
    @task
    def search_vector_db_for_a_movie(**context) -> None:
        """
        Gets the query string from the user, embeds it using the same embedding models used for the rest of the data,
        and then finds the nearest neighbor to it inside the vector database collection.

        returns: None - Displays the title, release date, and the description of the movie that is the closest to the
                        provided query.
        """
        import chromadb
        from fastembed import TextEmbedding

        query_str = context["params"]["query_string"]
        path = "./vectordb/"
        client = chromadb.PersistentClient(path=path)

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)
        collection = client.get_collection(COLLECTION_NAME)

        query_emb = list(embedding_model.embed([query_str]))[0]

        results = collection.query(
            query_embeddings=query_emb,
            n_results=1,
        )
        for result in results['metadatas'][0]:
            print(f"You should watch: {result['title']} released in {result['release_date']}")
            print("Description:")
            print(result["description"])

    search_vector_db_for_a_movie()


return_best_fit()