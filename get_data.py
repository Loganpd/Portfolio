from typing import List
from pendulum import datetime, duration
from airflow.sdk import chain, dag, task, Asset


COLLECTION_NAME = "movies"
MOVIE_DESCRIPTION_DIRECTORY = "./include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"


#%%
@dag(start_date=datetime(2025, 6, 15),
     schedule='@daily',
     default_args={
         "retries": 5,
         "retry_delay": duration(seconds=10)
     })
def get_data():
    """
    A DAG that consists of 5 tasks.
    1. Create a vector database collection if it does not already exist
    2. Load files containing the descriptions of movies
    3. Extract movie descriptions from the said files
    4. Create vector embeddings of the movie descriptions
    5. Load the vector embeddings into the vector database
    """
    #%%
    @task(retries=5, retry_delay=duration(seconds=2))
    def create_collection_if_not_exists() -> None:
        """
        Checks for existence of the collection with the given name, and then creates it if it does not already exist.

        returns: None - only displays information about the action taken in the logs.
        """
        import chromadb
        path = "./vectordb/"
        client = chromadb.PersistentClient()

        if COLLECTION_NAME not in [collection.name for collection in client.list_collections()]:
            print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
            collection = client.create_collection(COLLECTION_NAME)
            print(f"Collection {COLLECTION_NAME} created successfully.")
            print(f"Collection details: {collection}")
        else:
            print(f"Collection {COLLECTION_NAME} already exists.")

    _create_collection_if_not_exists = create_collection_if_not_exists()


    #%%
    @task
    def list_movie_description_files() -> list:
        """
        Gets the list of names of all the files containing the descriptions of movies. Normally, these descriptions
        are fetched using API calls to other systems but for the sake of demonstration, they are placed inside separate
        files and are loaded in this method.

        returns: a list of file names containing the descriptions of movies.
        """
        import os

        movie_description_files = [f for f in os.listdir(MOVIE_DESCRIPTION_DIRECTORY) if f.endswith('.txt')]
        return movie_description_files


    _list_movie_description_files = list_movie_description_files()

    #%%
    @task
    def transform_movie_description_file(movie_description_file: str) -> List[dict]:
        """
        Reads the files containing the descriptions of movies and parses movie titles, release dates, and descriptions.
        Next, it places the parsed information from the description files, into a list of dictionaries and passes it
        onto the next task.
        It is also designed in a manner that allows for multiple parallel runs of the same task at a given time.

        returns: a list of dictionaries containing the parsed information from the description files.
        """
        import os

        with open(os.path.join(MOVIE_DESCRIPTION_DIRECTORY, movie_description_file), "r") as f:
            movie_descriptions = f.readlines()
        titles = [movie_description.split("=")[0].strip() for movie_description in movie_descriptions]
        release_dates = [movie_description.split("=")[1].strip() for movie_description in movie_descriptions]
        movie_description_text = [movie_description.split("=")[2].strip() for movie_description in movie_descriptions]
        movie_descriptions = [
            {
                "title": title,
                "release_date": release_date,
                "description": description,
            }
            for title, release_date, description in zip(titles, release_dates, movie_description_text)
        ]
        return movie_descriptions


    _transformed_movie_description_files = transform_movie_description_file.expand(movie_description_file=_list_movie_description_files)

    #%%
    @task
    def create_vector_embeddings(movie_data: List[dict]):
        """
        Uses the previously defined embedding model in order to create vector embeddings for the description texts.

        returns: a list of movie description vector embeddings.
        """
        from fastembed import TextEmbedding
        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)
        movie_descriptions = [movie["description"] for movie in movie_data]
        description_embeddings = [
            list(map(float, next(embedding_model.embed([desc])))) for desc in movie_descriptions
        ]
        return description_embeddings


    _create_vector_embeddings = create_vector_embeddings.expand(movie_data=_transformed_movie_description_files)

    #%%
    @task(outlets=[Asset("movie_vector_data")], trigger_rule="all_done")
    def load_embeddings_to_vector_db(list_of_movie_data: list, list_of_description_embeddings: list) -> None:
        """
        Received the text embeddings and the movie descriptions and stores them in a chromadb collection for future use.

        returns: None - Only stores data inside the vector database collection.
        """
        import chromadb
        import uuid
        path = "./vectordb/"
        client = chromadb.PersistentClient(path=path)
        collection = client.get_collection(COLLECTION_NAME)

        ids = [str(uuid.uuid4()) for _ in range(len(list_of_movie_data))]
        collection.add(embeddings=list_of_description_embeddings, metadatas=list_of_movie_data, ids=ids)

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(list_of_movie_data=_transformed_movie_description_files,
                                                                 list_of_description_embeddings=_create_vector_embeddings,)

    chain(_create_collection_if_not_exists,_load_embeddings_to_vector_db)


get_data()