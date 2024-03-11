import os
import json
from typing import Any, List, Optional

from chromadb.api.types import ID, Document, Metadata, OneOrMany, QueryResult
import jedi
import chromadb

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma

from llmcoder.utils import get_data_dir


class Index:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.client = GPT4AllEmbeddings()
        self.db_dir = os.path.join(get_data_dir(), "database")
        self.database = chromadb.PersistentClient(self.db_dir)
        self.collection_name = "llmcoder"

    def create(self, code: str, collection_name: str = "llmcoder") -> None:
        self.collection_name = collection_name

        script = jedi.Script(code)
        reference_names = script.get_names(all_scopes=True, references=True, definitions=True)

        attribute_names = []
        attributes = []

        names = []
        metadatas = []
        embeddings = []
        ids = []
        i = 0

        for reference_name in reference_names:
            if (not reference_name.in_builtin_module()) and reference_name.type in ("module", "class", "interface"):
                defined_names = reference_name.defined_names()
                for defined_name in defined_names:
                    if defined_name.type == "function" and (defined_name.name not in attribute_names):
                        attribute_names.append(defined_name.name)
                        attributes.append({ 
                            "metadata": {
                                "module": defined_name.full_name.split(".")[0] if defined_name.full_name is not None else defined_name.name,
                                "full_name": defined_name.full_name,
                                "name": defined_name.name
                            }, "embedding": defined_name.name
                        })
                        embeddings.append(self.client.embed_query(defined_name.name))
                        names.append(defined_name.name)
                        metadatas.append({
                            "module": defined_name.full_name.split(".")[0] if defined_name.full_name is not None else defined_name.name,
                            "full_name": defined_name.full_name if defined_name.full_name is not None else defined_name.name,
                            "name": defined_name.name,
                         })
                        ids.append(str(i))
                        i = i + 1

                for defined_name in defined_names:
                    if (not defined_name.in_builtin_module()) and defined_name.type in ("module", "class", "interface"):
                        d_names = defined_name.defined_names()
                        for d_name in d_names:
                            if d_name.type == "function" and (d_name.name not in attribute_names):
                                attribute_names.append(d_name.name)
                                attributes.append({ 
                                    "metadata": {
                                        "module": d_name.full_name.split(".")[0] if d_name.full_name is not None else d_name.name,
                                        "full_name": d_name.full_name,
                                        "name": d_name.name
                                    }, "embedding": d_name.name
                                })
                                embeddings.append(self.client.embed_query(d_name.name))
                                names.append(d_name.name)
                                metadatas.append({
                                    "module": d_name.full_name.split(".")[0] if d_name.full_name is not None else d_name.name,
                                    "full_name": d_name.full_name if d_name.full_name is not None else d_name.name,
                                    "name": d_name.name,
                                 })
                                ids.append(str(i))
                                i = i + 1

        with open("./temp-attributes.jsonl", "w") as jsonfile:
            json.dump(attributes, jsonfile)

        self.index(names, embeddings, metadatas, ids)

    def index(self, names: List[str], embeddings: Any, metadatas: OneOrMany[Metadata], ids: OneOrMany[ID]) -> str:
        self.collection = self.database.get_or_create_collection(name=self.collection_name)
        self.collection.upsert(documents=names, embeddings=embeddings, metadatas=metadatas, ids=ids)
        return self.collection_name

    def query(self, attribute: str, top_k: int = 10, module: Optional[str] = None, full_name: Optional[str] = None, collection_name: str = "llmcoder") -> Any:
        self.collection_name = collection_name
        self.collection = self.database.get_or_create_collection(name=self.collection_name)
        embeddings = self.client.embed_query(attribute)

        if module is None and full_name is None:
            return self.collection.query(query_embeddings=embeddings, n_results=top_k)['documents'][0]
        if module is not None and full_name is None:
            return self.collection.query(query_embeddings=embeddings, where={"module": module}, n_results=top_k)['documents'][0]
        if module is None and full_name is not None:
            return self.collection.query(query_embeddings=embeddings, where={"full_name": full_name}, n_results=top_k)['documents'][0]
        if module is not None and full_name is not None:
            return self.collection.query(query_embeddings=embeddings, where={"full_name": full_name, "module": module}, n_results=top_k)['documents'][0]

        return None
