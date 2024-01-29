from haystack.document_stores import InMemoryDocumentStore
from pathlib import Path
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever, PDFToTextConverter
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from pprint import pprint
from haystack import Pipeline
from haystack.utils import print_answers


converter = PDFToTextConverter()

doc_dir  = "./ww2.pdf"
document_store = InMemoryDocumentStore(use_bm25=True)
indexing_pipeline = Pipeline()
indexing_pipeline.add_node(component=converter, name="PDFConverter", inputs=["File"])
indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PDFConverter"])
indexing_pipeline.run_batch(file_paths=[doc_dir])
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="timpal0l/mdeberta-v3-base-squad2", use_gpu=True)
pipe = ExtractiveQAPipeline(reader, retriever=retriever)

prediction = pipe.run(
    query = "Хитлер хэзээ амиа хорлосон бэ?", params = {"Retriever": {"top_k":10}, "Reader": {"top_k":5}}
)


print_answers(prediction, details="minimum")