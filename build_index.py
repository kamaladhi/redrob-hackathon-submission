import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm

def is_honeypot(candidate):
    # Heuristic 1: Expert skill with 0 months experience
    for skill in candidate.get("skills", []):
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 1) == 0:
            return True
    
    # Heuristic 2: Impossible years of experience
    # For example, if the sum of career history duration is drastically less than years_of_experience
    total_career_months = sum(job.get("duration_months", 0) for job in candidate.get("career_history", []))
    stated_yoe_months = candidate.get("profile", {}).get("years_of_experience", 0) * 12
    # allow some discrepancy but if stated is > 20 years and career is 0 months
    if stated_yoe_months > 60 and total_career_months < 12:
        # Might be a honeypot, but let's be conservative
        pass 
        
    return False

def build_candidate_text(candidate):
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "")
    summary = profile.get("summary", "")
    
    skills = [s.get("name", "") for s in candidate.get("skills", [])]
    skills_text = ", ".join(skills)
    
    # Career history descriptions
    career_desc = " ".join([job.get("description", "") for job in candidate.get("career_history", [])])
    
    return f"Title: {title}. Summary: {summary}. Skills: {skills_text}. Experience: {career_desc}"

def main():
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    in_file = r'd:\Hackathons\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'
    
    candidates = []
    texts = []
    metadata = []
    
    print("Parsing candidates...")
    with open(in_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            if not line.strip(): continue
            cand = json.loads(line)
            candidates.append(cand)
            
            # Detect honeypots
            honeypot_flag = is_honeypot(cand)
            
            text = build_candidate_text(cand)
            texts.append(text)
            
            # Extract structured features for hybrid scoring
            profile = cand.get("profile", {})
            signals = cand.get("redrob_signals", {})
            
            # Key skills to check
            skills = [s.get("name", "").lower() for s in cand.get("skills", [])]
            has_python = "python" in skills
            has_vector_db = any(db in skills for db in ["faiss", "pinecone", "weaviate", "qdrant", "milvus"])
            has_embeddings = any(kw in skills for kw in ["embeddings", "sentence-transformers", "nlp"])
            
            meta = {
                "candidate_id": cand["candidate_id"],
                "is_honeypot": honeypot_flag,
                "years_of_experience": profile.get("years_of_experience", 0),
                "has_python": has_python,
                "has_vector_db": has_vector_db,
                "has_embeddings": has_embeddings,
                "recruiter_response_rate": signals.get("recruiter_response_rate", 0.0),
                "notice_period_days": signals.get("notice_period_days", 90),
                "last_active_date": signals.get("last_active_date", "2000-01-01"),
                "interview_completion_rate": signals.get("interview_completion_rate", 0.0),
                "candidate_json": json.dumps(cand) # Store full JSON for reasoning extraction later
            }
            metadata.append(meta)
            
    print("Generating embeddings...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner product since MiniLM embeddings are often normalized, but let's normalize to use cosine
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    faiss.write_index(index, "candidate_index.faiss")
    
    print("Saving metadata...")
    df = pd.DataFrame(metadata)
    df.to_pickle("candidate_metadata.pkl")
    
    print("Done!")

if __name__ == "__main__":
    main()
