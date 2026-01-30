import os
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from ragas.run_config import RunConfig
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Configure Pandas for clean terminal reports
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 50)

def run_evaluation():
    """
    Runs an offline test assessment of the Lucknow Guide RAG layer using active Groq models.
    """
    print("\n--- Initializing RAG Evaluation Suite ---")

    # 1. Standardize Embedding Model Configuration
    embedding_model = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model="BAAI/bge-small-en-v1.5"
    )
    
    # 2. Point to the ACTUAL persistent database directory used by agent_logic.py
    persist_dir = "./chroma_db_lucknow"
    print(f"1. Loading reference vector database from: {persist_dir}...")
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    else:
        print(f"❌ Error: Real vector store directory '{persist_dir}' not found or empty.")
        print("Please boot up your main server first so that agent_logic.py generates the database structures.")
        return

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Setup Groq Models (Using Active Production Identifiers)
    print("2. Connecting to Groq Inference Engines...")

    # 1. Initialize LLM
    # llm = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.5)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5)
    # llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.5)
    model="meta-llama/llama-4-scout-17b-16e-instruct"
    
    # Active production model used to generate answers for evaluation questions
    qa_llm = ChatGroq(
        model_name=model, 
        temperature=0, 
        max_retries=3
    )

    # High-reasoning model acting as the objective evaluation judge
    ragas_llm = ChatGroq(
        model_name=model,
        temperature=0,
        max_retries=3,
        timeout=120  # Safeguards against API congestion rate limit timeouts
    )

    # 4. Build standard QA validation pipeline
    qa_chain = RetrievalQA.from_chain_type(llm=qa_llm, chain_type="stuff", retriever=retriever)

    # 5. Define Controlled Benchmark Evaluation Dataset
    eval_questions = [
        "What is the historical significance of the Bara Imambara?",
        "Describe the famous Galouti Kebab.",
        "Who was Asaf-ud-Daula?",
        "What makes the Tunday Kababi so special according to the text?",
        "How is Lucknowi Biryani different from other types?",
    ]
    ground_truths = [
        "The Bara Imambara is a significant historical monument built in 1784 by Asaf-ud-Daula, known for its large central hall built without beams and a food-for-work program.",
        "The Galouti Kebab is a melt-in-the-mouth kebab made from finely minced meat and over 100 spices, created for a toothless Nawab.",
        "Asaf-ud-Daula was the fourth Nawab of Awadh who commissioned projects like the Bara Imambara to provide employment during a famine.",
        "According to the text, Tunday Kababi is special because of its secret family recipe using over 160 spices.",
        "Lucknowi Biryani is different because it is less spicy and more subtle and fragrant. The meat and rice are cooked separately before being layered and slow-cooked in the 'dum style'."
    ]

    print("\n--- Generating System Outputs for Evaluation Dataset ---")
    answers = []
    contexts = []
    
    for question in eval_questions:
        print(f"    -> Testing Question: '{question}'")
        try:
            response = qa_chain.invoke(question)
            answers.append(response["result"])
            
            retrieved_docs = retriever.invoke(question)
            contexts.append([doc.page_content for doc in retrieved_docs])
        except Exception as e:
            print(f"    ❌ Error processing question: {question}. Details: {e}")
            answers.append("Execution Timeout/Failure Error")
            contexts.append([])

    # 6. Parse Arrays into Ragas Datasets
    dataset_dict = {
        "question": eval_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(dataset_dict)

    print("\n--- Instructing LLM Judge to Score Metrics ---")
    metrics = [
        faithfulness,
        context_precision,
        context_recall
    ]
    
    # Run evaluation serially to stay within Groq free-tier rate limitations safely
    config = RunConfig(max_workers=1)

    result = evaluate(
        dataset=dataset, 
        metrics=metrics, 
        llm=ragas_llm,  
        embeddings=embedding_model,  
        run_config=config,  
        raise_exceptions=False 
    )

    print("\n--- Evaluation Processing Complete ---")
    df = result.to_pandas()
    
    # Save formatted reports to workspace root
    html_report_path = "evaluation_report.html"
    md_report_path = "evaluation_report.md"
    
    try:
        df.to_html(html_report_path, index=False, border=1, classes='table table-striped table-hover')
        print(f"✅ Detailed HTML scorecard generated: {html_report_path}")
    except Exception as e:
        print(f"Could not export HTML report format: {e}")

    try:
        df.to_markdown(md_report_path, index=False)
        print(f"✅ Detailed Markdown scorecard generated: {md_report_path}")
    except Exception as e:
        print(f"Could not export Markdown report format: {e}")

    print("\n================= RAG METRICS PERFORMANCE VIEW =================")
    print(df.to_string())


if __name__ == "__main__":
    run_evaluation()