from ..factories.connection_string_factory import create_connection_string
from ..factories.embedding_factory import create_embedding
import config
import os

from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.vectorstores.pgvector import DistanceStrategy


def upload(upload_file, collection_name):

    file_path = './tmp/' + upload_file.name
    try:
        # Files in Django will be splited into multiple chunk, we shall merge them together
        with open(file_path, 'wb+') as destination:
            for chunk in upload_file.chunks():
                destination.write(chunk)
        print(file_path)
        file_extension = os.path.splitext(
            file_path)[1][1:]  # get the file_extension

        if file_extension == "csv":
            loader = CSVLoader(file_path=file_path)
            docs = loader.load_and_split()  # return a list of Document Objects

        elif file_extension == "pdf":
            loader = PyPDFLoader(file_path=file_path)
            docs = loader.load_and_split()  # return a list of Document Objects

        elif file_extension == "txt":
            print("reading txt file")
            loader = TextLoader(file_path=file_path)
            docs = loader.load_and_split()  # return a list of Document Objects
            print(type(docs))
        else:
            raise ValueError("Can't support " +
                             file_extension + " type, I'm sorry.")
    except ValueError as e:
        print(f"An error occurred: {str(e)}")

    embeddings = create_embedding("OpenAI")  # Creating embedding method
    connection_string = create_connection_string(
        "postgres")  # Creating Postgres connection string

    database = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=connection_string,
        distance_strategy=DistanceStrategy.COSINE,
        openai_api_key=config.OPENAI_API_KEY,
        pre_delete_collection=True,
    )

    # Deleting the file in tmp, since it already exist in Database
    if os.path.isfile(file_path):
        os.remove(file_path)

    return {"result": "success"}
