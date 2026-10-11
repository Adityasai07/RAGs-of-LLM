from typing import List, Dict, Any
from langchain_groq import ChatGroq

class RAGpipeline:
    def __init__(self, rag_retriver, llm: ChatGroq):
        self.rag_retriver = rag_retriver
        self.llm = llm
        self.history: List[Dict[str, Any]] = []

    def query(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.2,
        stream: bool = False,
        summarize: bool = False
    ) -> Dict[str, Any]:
        results = self.rag_retriver.retrive(query, top_k, min_score)
        if not results:
            answer = "No relevant context found."
            sources = []
        else:
            context = "\n\n".join([doc["content"] for doc in results])
            sources = [{
                "source": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
                "page": doc["metadata"].get("page", "N/A"),
                "file_type": doc["metadata"].get("file_type", "unknown"),
                "score": doc["similarity_score"],
                "preview": doc["content"][:120] + "..."
            } for doc in results]

            prompt = (
                f"Use the following context to answer the question concisely.\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )

            if stream:
                print("Answer: ", end="", flush=True)
                streamed_tokens = []
                for chunk in self.llm.stream(prompt):
                    token = chunk.content
                    print(token, end="", flush=True)
                    streamed_tokens.append(token)
                print("\n")
                answer = "".join(streamed_tokens)
            else:
                response = self.llm.invoke(prompt)
                answer = response.content

        citations = [
            f"[{i+1}] {src['source']} ({src['file_type']}"
            + (f", page {src['page']}" if src.get("page") not in (None, "N/A", "unknown") else "")
            + ")"
            for i, src in enumerate(sources)
        ]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke(summary_prompt)
            summary = summary_resp.content

        record = {
            "query": query,
            "answer": answer_with_citations,
            "sources": sources,
            "summary": summary
        }
        self.history.append(record)

        return {
            "query": query,
            "answer": answer_with_citations,
            "sources": sources,
            "summary": summary,
            "history": self.history
        }