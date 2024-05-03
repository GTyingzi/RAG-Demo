#!/usr/bin/env python.txt
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/5/3 12:52 
# ide： PyCharm
import os
import argparse
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from tqdm import tqdm

# 功能：解析 txt 文件，并按行分割成段
def process_file(file_path):
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
        sentences = text.split('\n')
        return text, sentences

# 构建Prompt
def generate_rag_prompt(data_point):
    return f"""### Instruction
        {data_point["instruction"]}
        ### Input:
        {data_point["input"]}
        ### Response
    """


# 文档Embedder类
class DocumentEmbedder:
    def __init__(
            self,
            model_name='BAAI/bge-large-zh-v1.5',
            max_length=128,
            max_number_of_sentences=20
    ):
        # Load AutoModel from huggingface model repository
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # this parameter dictates the maximum number of tokens per sentence
        self.max_length = max_length
        # This dictates the maximum number of sentences to be considerd
        self.max_number_of_sentences = max_number_of_sentences

    def get_document_embeddings(self, sentences):
        # Keep only the first K sentences for GPU purpose
        sentences = sentences[:self.max_number_of_sentences]
        # Tokenizer the sentences
        encoded_input = self.tokenizer(sentences,
                                       padding=True,
                                       truncation=True,
                                       max_length=128,
                                       return_tensors="pt"
                                       )
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # The Document's embedding is the average of all sentences
        # If there's only one sentence, then it's just it's embedding
        return torch.mean(model_output.pooler_output, dim=0, keepdim=True)


class GenerativeModel:
    def __init__(
            self,
            model_path="Qwen/Qwen1.5-0.5B",
            max_input_length=200,
            max_generated_length=200
    ):
        # Load 4-bit quantized model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length = max_input_length
        self.max_generated_length = max_generated_length
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def answer_prompt(self, prompt):
        # Tokenizer the sentences
        encoded_input = self.tokenizer([prompt],
                                       padding=True,
                                       truncation=True,
                                       max_length=self.max_input_length,
                                       return_tensors="pt"
                                       )
        outputs = self.model.generate(input_ids=encoded_input['input_ids'].to(self.device),
                                      attention_mask=encoded_input['attention_mask'].to(self.device),
                                      max_new_tokens=self.max_generated_length,
                                      do_sample=False
                                      )
        decoder_text = self.tokenizer.batch_decode(outputs,
                                                   skip_special_tokens=True
                                                   )
        return decoder_text

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents_directory", help="The directory that has the documents", default='rag_documents')
    parser.add_argument("--embedding_model", help="The HuggingFace path to the embedding model to use", default="AI-ModelScope/bge-large-zh")
    parser.add_argument("--generative_model", help="The HuggingFace path to the generative model to use", default="qwen/Qwen1.5-0.5B")
    parser.add_argument("--number_of_documents", help="The number of relevant documents to use for context", default=2, type=int)
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()

    print("Splitting documents into sentences")
    documents = []
    for idx, file in enumerate(tqdm(os.listdir(args.documents_directory)[:10])):
        # 如果file是文件夹，跳过
        if os.path.isdir(os.path.join(args.documents_directory, file)):
            continue
        current_file_path = os.path.join(args.documents_directory, file)
        text, sentences = process_file(current_file_path)
        document = {
            "file_path": file,
            "document_text": text,
            "sentences": sentences
        }
        documents.append(document)

    print("Loading Document Embedder")
    document_embedder = DocumentEmbedder(model_name=args.embedding_model, max_length=128, max_number_of_sentences=20)
    embeddings = []
    for document in tqdm(documents):
        embeddings.append(document_embedder.get_document_embeddings(document["sentences"]))
    embeddings = torch.concat(embeddings, dim=0).data.cpu().numpy()
    embedding_dimensions = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(int(embedding_dimensions))
    faiss_index.add(embeddings)

    question = "c++是一种什么样的语言"
    query_embedding = document_embedder.get_document_embeddings([question])
    distances, indices = faiss_index.search(query_embedding.data.cpu().numpy(), args.number_of_documents)

    context = ''
    for idx in indices[0]:
        context += documents[idx]["document_text"]

    rag_prompt = generate_rag_prompt({
        "instruction": question,
        "input": context
    })

    print("Generating Answer")
    generative_model = GenerativeModel(model_path=args.generative_model, max_input_length=200, max_generated_length=200)
    answer = generative_model.answer_prompt(rag_prompt)[0].split("### Response")[0]
    print(answer)
