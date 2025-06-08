

import os
from typing import List, Dict, Tuple
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

class AgentSystem:
    _instance = None # Singleton pattern to ensure only one instance of the model is loaded

    def __new__(cls):
        """Ensures that only one instance of AgentSystem is created."""
        if cls._instance is None:
            cls._instance = super(AgentSystem, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initializes LangChain components like LLM, embeddings, vector store, and agents."""
        print("Initializing AgentSystem (LangChain components)...")

        # --- LLM Setup ---
        # Checks for OpenAI API key. Uses a dummy LLM if not found for demonstration.
        if "OPENAI_API_KEY" not in os.environ:
            print("WARNING: OPENAI_API_KEY not found"
                  "For full functionality, please set OPENAI_API_KEY environment variable.")
            class DummyChatOpenAI: # Mock LLM for local testing without an API key
                def __init__(self, model_name="dummy", temperature=0.7):
                    self.model_name = model_name
                    self.temperature = temperature

                def invoke(self, prompt_messages: List[BaseMessage]) -> BaseMessage:
                    prompt_text = "\n".join([msg.content for msg in prompt_messages if isinstance(msg, HumanMessage)])
                    if "summarize" in prompt_text.lower():
                        return HumanMessage(content="This is a concise summary from the dummy LLM, focusing on key aspects related to the query from dummy data.")
                    elif "answer the question" in prompt_text.lower():
                        return HumanMessage(content="Based on the dummy information, the answer is: a generic response derived from simulated data.")
                    elif "generate citations" in prompt_text.lower():
                        return HumanMessage(content="Source: dummy_doc_A.pdf, Source: dummy_doc_B.pdf.")
                    else:
                        return HumanMessage(content="This is a general response from the Dummy LLM.")
            self.llm = DummyChatOpenAI()
        else:
    
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7) 

        # --- Embedding Model & Text Splitter ---
      
        self.embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        # --- PDF Document Loading and Processing ---
        self._load_and_process_pdfs() 
        
        self._setup_tools()
        self._setup_agent_executor()
        print("AgentSystem initialized successfully.")


    def _load_and_process_pdfs(self):
        """Loads PDF documents from the 'docs/' directory and sets up the vector store."""
        pdf_directory = "docs/"
        print(f"Attempting to load PDF documents from '{pdf_directory}'...")
        
        # Ensure the directory exists
        if not os.path.exists(pdf_directory):
            print(f"Error: Directory '{pdf_directory}' not found. Please create it and place your PDFs inside.")
            self.vectorstore = None 
            self.retriever = None
            return

        # Get all PDF file paths in the directory
        pdf_paths = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_paths:
            print(f"No PDF files found in '{pdf_directory}'. Please add some PDF documents to this folder for the system to process.")
            self.vectorstore = None
            self.retriever = None
            return

        all_langchain_docs = []
        for pdf_path in pdf_paths:
            try:
                loader = PyPDFLoader(pdf_path)
                # Load and split documents from the current PDF
                docs = loader.load_and_split(self.text_splitter)
                for doc in docs:
                    # Add source metadata (filename) to each chunk
                    doc.metadata["source"] = os.path.basename(pdf_path)
                all_langchain_docs.extend(docs)
                print(f"Loaded and processed '{os.path.basename(pdf_path)}'. Total chunks so far: {len(all_langchain_docs)}")
            except Exception as e:
                print(f"Error loading '{os.path.basename(pdf_path)}': {e}")
                # If an error occurs, continue to the next PDF to avoid stopping the whole process

        if not all_langchain_docs:
            print("No documents were successfully loaded or processed. Vector store will be empty.")
            self.vectorstore = None
            self.retriever = None
            return

        # Create an in-memory FAISS vector store from all processed chunks
        self.vectorstore = FAISS.from_documents(all_langchain_docs, self.embedding_model)
        # Configure the retriever to fetch top 3 most relevant chunks (can adjust k)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        print(f"Vector store created with {len(all_langchain_docs)} chunks from PDF(s).")

    def _setup_tools(self):
        """Defines the specialized agents as LangChain tools."""

        @tool
        def retrieve_documents(query: str) -> List[Dict]:
            """
            Fetches relevant document chunks based on the user's query from loaded PDFs.
            Returns a list of dictionaries, each containing 'chunk_text', 'source' (filename), and 'chunk_id'.
            """
            instance = AgentSystem._instance # Access the singleton instance
            if instance.retriever is None:
                # Handle case where no documents were loaded in _load_and_process_pdfs
                return [{"chunk_text": "No documents loaded for retrieval. Please ensure PDFs are in the 'docs/' directory.", "source": "system_error", "chunk_id": "error_no_docs"}]

            print(f"\n--- Tool: Retriever Agent called for query: '{query}' ---")
            docs = instance.retriever.invoke(query)
            retrieved_info = []
            for doc in docs:
                # 'source' will be the filename from PDF metadata, 'chunk_id' is for internal tracking
                retrieved_info.append({
                    'chunk_text': doc.page_content,
                    'source': doc.metadata.get('source', 'unknown_source_pdf'),
                    'chunk_id': doc.metadata.get('chunk_id', 'unknown_chunk_id')
                })
            print(f"Retrieved {len(retrieved_info)} chunks.")
            return retrieved_info

        @tool
        def summarize_text(text_chunks: List[Dict], query: str) -> str:
            """
            Condenses a list of text chunks into a concise summary relevant to the user's query.
            Takes a list of dictionaries (from retrieve_documents) and the original query.
            """
            instance = AgentSystem._instance # Access the singleton instance
            print(f"\n--- Tool: Summarizer Agent called for {len(text_chunks)} chunks and query: '{query}' ---")
            # Check if retrieval returned an error message (no documents loaded/found)
            if not text_chunks or text_chunks[0].get("source") == "system_error":
                return "No relevant text available to summarize due to document loading or retrieval issues."

            combined_text = "\n\n".join([chunk['chunk_text'] for chunk in text_chunks])

            summarize_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Summarize the following text concisely, focusing on information relevant to the user's query."),
                ("human", "User Query: {query}\n\nText to summarize:\n{text}\n\nConcise Summary:")
            ])
            summarize_chain = summarize_prompt | instance.llm | StrOutputParser()
            summary = summarize_chain.invoke({"query": query, "text": combined_text})
            print("Summary generated.")
            return summary

        @tool
        def answer_question(summarized_text: str, query: str) -> str:
            """
            Answers the user's question based solely on the provided summarized text.
            Ensures the answer is factual and directly supported by the summary.
            """
            instance = AgentSystem._instance # Access the singleton instance
            print(f"\n--- Tool: QA Agent called for query based on summary ---")
            if summarized_text == "No relevant text available to summarize due to document loading or retrieval issues.":
                return "I cannot answer the question as no relevant information was found or summarized from the documents."

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a factual question-answering assistant. Answer the user's question using ONLY the provided summarized information. Do not use outside knowledge or make assumptions."),
                ("human", "Question: {query}\n\nSummarized Information:\n{summary}\n\nAnswer:")
            ])
            qa_chain = qa_prompt | instance.llm | StrOutputParser()
            answer = qa_chain.invoke({"query": query, "summary": summarized_text})
            print("Answer generated.")
            return answer

        @tool
        def generate_citations(original_chunks: List[Dict], summarized_text: str, answer: str) -> List[str]:
            """
            Generates citations for the information provided in the answer and summary.
            This tool examines the original chunks used and provides their sources (filenames).
            It's called by the main agent to provide references.
            """
            print(f"\n--- Tool: Citation Agent called to generate citations ---")
            # Check if retrieval returned an error message
            if not original_chunks or original_chunks[0].get("source") == "system_error":
                return ["No specific citations available due to document loading or retrieval issues."]
            
            cited_sources = set()
            for chunk in original_chunks:
                cited_sources.add(chunk['source'])
            citations = [f"Source: {doc_id}" for doc_id in sorted(list(cited_sources))]
            print(f"Citations generated: {citations}")
            return citations

        self.tools = [retrieve_documents, summarize_text, answer_question, generate_citations]

    def _setup_agent_executor(self):
        """Sets up the main LangChain AgentExecutor."""
        # The system prompt guides the main agent's behavior and tool usage.
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a helpful and intelligent assistant designed to answer user questions based on provided PDF documents.
                You have access to the following tools: retrieve_documents, summarize_text, answer_question, and generate_citations.
                
                Here's your workflow:
                1. First, always use `retrieve_documents` to find relevant information for the user's query from the PDFs.
                2. Then, use `summarize_text` on the retrieved documents, focusing on the user's query.
                3. Next, use `answer_question` to provide a concise answer based ONLY on the summarized text.
                4. Finally, use `generate_citations` with the original retrieved chunks, the summary, and the answer, to provide source references.
                5. Your final response should be a well-formatted string that clearly presents the Answer, Summary, and Citations.
                   Structure it as:
                   Answer: [Your answer here]
                   Summary: [Your summary here]
                   Citations: [List of citations here, e.g., Source: document1.pdf, Source: document2.pdf]
                
                Always try to follow this workflow to provide a comprehensive answer with references. If a tool call fails or no information is found after retrieval, state that you couldn't find relevant information in the documents."""),
                MessagesPlaceholder(variable_name="chat_history", optional=True), # For conversational memory if needed
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"), # For agent's internal thought process
            ]
        )

        # Creates an agent that leverages OpenAI's function calling capabilities for tool usage.
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)

        # AgentExecutor is the runtime for the agent, managing its thought process and tool calls.
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True, # Set to True to see the agent's detailed thought process
            handle_parsing_errors=True # Good for production to gracefully handle malformed outputs
        )

    async def query_agent(self, user_query: str) -> Dict:
        """
        Invokes the LangChain AgentExecutor with the user's query.
        Returns a dictionary containing the structured response (answer, summary, citations).
        """
        print(f"\n===== Invoking AgentExecutor for query: '{user_query}' =====")
        if self.retriever is None:
            # This state means no PDFs were loaded successfully during startup.
            return {
                "answer": "The system is not initialized because no PDF documents were found or processed. Please ensure sports-related PDFs are in the 'docs/' directory.",
                "summary": "N/A",
                "citations": ["N/A"]
            }
        try:
            # The agent executor will manage the calls to retrieve, summarize, answer, and cite.
            result = await self.agent_executor.ainvoke({"input": user_query, "chat_history": []})

            final_output = result.get('output', "No clear answer could be generated by the agent.")
            
            # Simple parsing of the expected output format from the LLM's final output
            answer_prefix = "Answer: "
            summary_prefix = "Summary: "
            citations_prefix = "Citations: "

            parsed_answer = "Could not extract answer or no relevant information found."
            parsed_summary = "Could not extract summary or no relevant information found."
            parsed_citations = []

            # Using find() to locate prefixes and extract content
            answer_start_idx = final_output.find(answer_prefix)
            summary_start_idx = final_output.find(summary_prefix)
            citations_start_idx = final_output.find(citations_prefix)

            if answer_start_idx != -1:
                answer_end_idx = summary_start_idx if summary_start_idx != -1 else citations_start_idx if citations_start_idx != -1 else len(final_output)
                parsed_answer = final_output[answer_start_idx + len(answer_prefix):answer_end_idx].strip()

            if summary_start_idx != -1:
                summary_end_idx = citations_start_idx if citations_start_idx != -1 else len(final_output)
                parsed_summary = final_output[summary_start_idx + len(summary_prefix):summary_end_idx].strip()

            if citations_start_idx != -1:
                citations_text = final_output[citations_start_idx + len(citations_prefix):].strip()
                # Assuming citations are comma-separated
                parsed_citations = [c.strip() for c in citations_text.split(',') if c.strip()] 

            print(f"\n===== Agent Execution Complete =====")
            return {
                "answer": parsed_answer,
                "summary": parsed_summary,
                "citations": parsed_citations
            }
        except Exception as e:
            print(f"Error during agent execution: {e}")
            return {
                "answer": f"An error occurred while processing your query: {e}. Please check the server logs for more details.",
                "summary": "N/A",
                "citations": ["N/A"]
            }

agent_system_instance = AgentSystem()