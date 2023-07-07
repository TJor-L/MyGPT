import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from dotenv import load_dotenv
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

prompt_template = """
你现在是一名中国的法律专家。现在下面有一些关于中国宪法的信息。随后你将遇到一个问题，请你如实回答。如果下面的信息中没有问题的答案，请你不要编造回答，有礼貌地说自己不知道。
信息：{context}
 
问题: {question}"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

os.environ["OPENAI_API_KEY"] = "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"

embeddings = OpenAIEmbeddings()

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGVECTOR_HOST", "172.28.30.52"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE", "testdb"),
    user=os.environ.get("PGVECTOR_USER", "test"),
    password=os.environ.get("PGVECTOR_PASSWORD", "test"),
)
vectordb = PGVector.from_existing_index(
    embedding=embeddings,
    collection_name="constitution",
    distance_strategy=DistanceStrategy.COSINE,
    pre_delete_collection=False,
    connection_string=CONNECTION_STRING,
)

qa_chain = load_qa_chain(
    OpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", prompt=PROMPT)

qa = RetrievalQA(combine_documents_chain=qa_chain,
                 retriever=vectordb.as_retriever())


def query(input):
    print(input)
    output = qa.run(input)
    print(output)
    return output
