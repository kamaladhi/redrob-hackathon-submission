import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import argparse
import os

# JD text for semantic search
JD_TEXT = """Senior AI Engineer — Founding Team. Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning. 
Scrappy product-engineering attitude. Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users. 
Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS. Strong Python."""

def calculate_yoe_score(yoe):
    # Peak at 5-9 years.
    if 5 <= yoe <= 9:
        return 1.0
    elif 3 <= yoe < 5:
        return 0.8
    elif 9 < yoe <= 12:
        return 0.9
    elif yoe > 12:
        return 0.7
    else:
        return 0.4

def generate_reasoning(cand, yoe, has_python, has_vector_db, response_rate, notice_period):
    profile = cand.get("profile", {})
    title = profile.get("current_title", "Engineer")
    
    reasons = []
    reasons.append(f"{yoe} years of experience with current role as {title}.")
    
    tech_skills = []
    if has_python: tech_skills.append("Python")
    if has_vector_db: tech_skills.append("Vector DBs/Retrieval")
    
    if tech_skills:
        reasons.append(f"Strong background in {' and '.join(tech_skills)}.")
        
    if response_rate > 0.5:
        reasons.append(f"Excellent recruiter response rate ({int(response_rate*100)}%).")
    elif response_rate < 0.2:
        reasons.append(f"However, recruiter response rate is low ({int(response_rate*100)}%).")
        
    if notice_period > 60:
        reasons.append(f"Concern: notice period is long ({notice_period} days).")
        
    return " ".join(reasons)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to output CSV")
    args = parser.parse_args()
    
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding JD...")
    jd_embedding = model.encode([JD_TEXT])
    faiss.normalize_L2(jd_embedding)
    
    print("Loading index and metadata...")
    # NOTE: In a real submission, we might need to point to absolute paths relative to repo root
    # For now, we load from current working directory
    index = faiss.read_index("candidate_index.faiss")
    metadata_df = pd.read_pickle("candidate_metadata.pkl")
    
    print("Searching candidates...")
    # Get top 2000 from semantic search to refine
    k = min(2000, len(metadata_df))
    distances, indices = index.search(jd_embedding, k)
    
    scored_candidates = []
    
    for rank_idx, cand_idx in enumerate(indices[0]):
        meta = metadata_df.iloc[cand_idx]
        
        # Base semantic similarity score
        semantic_score = float(distances[0][rank_idx])
        
        # Hard filter for honeypots
        if meta["is_honeypot"]:
            continue
            
        yoe = meta["years_of_experience"]
        yoe_multiplier = calculate_yoe_score(yoe)
        
        # Tech skill multipliers
        tech_multiplier = 1.0
        if meta["has_python"]: tech_multiplier += 0.1
        if meta["has_vector_db"]: tech_multiplier += 0.2
        if meta["has_embeddings"]: tech_multiplier += 0.2
        
        # Behavioral multiplier
        # Notice period: 0-30 days is best (1.0), 30-60 is okay (0.9), >60 is penalty (0.8)
        notice_days = meta["notice_period_days"]
        notice_multiplier = 1.0 if notice_days <= 30 else (0.9 if notice_days <= 60 else 0.8)
        
        response_rate = meta["recruiter_response_rate"]
        response_multiplier = 0.5 + (0.5 * response_rate) # Maps 0->0.5, 1.0->1.0
        
        cand_json = json.loads(meta["candidate_json"])
        
        # Title penalty: penalize non-tech/unrelated titles heavily
        title_lower = cand_json.get("profile", {}).get("current_title", "").lower()
        title_multiplier = 1.0
        good_titles = ["ai", "ml", "machine learning", "data", "software", "backend", "engineer", "developer", "scientist"]
        bad_titles = ["mechanical", "civil", "accountant", "hr", "sales", "marketing", "operations", "designer", "graphic", "manager"]
        
        if any(bad in title_lower for bad in bad_titles) and not any(good in title_lower for good in good_titles):
            title_multiplier = 0.1 # Strong penalty for unrelated fields
        elif any(good in title_lower for good in ["ai", "ml", "machine learning"]):
            title_multiplier = 1.2 # Bonus for exact domain match
            
        # Calculate final score
        final_score = (semantic_score * 0.5) * yoe_multiplier * tech_multiplier * notice_multiplier * response_multiplier * title_multiplier
        
        reasoning = generate_reasoning(
            cand=cand_json, 
            yoe=yoe, 
            has_python=meta["has_python"], 
            has_vector_db=meta["has_vector_db"], 
            response_rate=response_rate, 
            notice_period=notice_days
        )
        
        scored_candidates.append({
            "candidate_id": meta["candidate_id"],
            "score": round(final_score, 4), # Round early so Python sorts on the exact values in the CSV
            "reasoning": reasoning
        })
        
    print("Sorting and generating submission...")
    # Sort by score descending, then candidate_id ascending for tie-breaks
    scored_candidates.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    # Take top 100
    top_100 = scored_candidates[:100]
    
    # Assign ranks
    final_output = []
    for rank, cand in enumerate(top_100, 1):
        final_output.append({
            "candidate_id": cand["candidate_id"],
            "rank": rank,
            "score": cand["score"],
            "reasoning": cand["reasoning"]
        })
        
    out_df = pd.DataFrame(final_output)
    out_df.to_csv(args.out, index=False)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
