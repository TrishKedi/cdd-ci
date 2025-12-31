
from core.services import CodeAnalyzer, Embedder, SimilarityLookup


code_analyzer = CodeAnalyzer()
embedder = Embedder()
similarity_lookup = SimilarityLookup()



def run_ci_pipeline():

    # for chunk in code_analyzer.extract_code_chuncks():
    #     pass
        
        # print(f"\n{chunk}\n")

    query_chunks = code_analyzer.extract_code_chuncks()


    if query_chunks:
        query = code_analyzer.get_processed_code(query_chunks)
        print("\n=======QUERY CHUNKS========\n")
        # print(f"\n{query}\n")
        print(f"\n{len(query)}\n")
        candidate_chunks = code_analyzer.extract_code_chuncks(query_files=False)
        candidate = code_analyzer.get_processed_code(candidate_chunks)
        print("\n=======CANDIDATE CHUNKS========\n")
        # print(f"\n{candidate}\n")
        print(f"\n{len(candidate)}\n")

        
        query_embeddings = embedder.generate_embeddings(query)
        candidate_embeddings = embedder.generate_embeddings(candidate)
        similarity_lookup.build_index(candidate_embeddings)

        search_results = similarity_lookup.run_semantic_search(query_embeddings)

        all_query_chunks = code_analyzer.get_processed_code(query_chunks, all=True)
        print(len(all_query_chunks))
        all_candidate_chunks = code_analyzer.get_processed_code(candidate_chunks, all=True)

        clones = similarity_lookup.process_results(
            search_results, 
            all_query_chunks, 
            all_candidate_chunks
        )

        print("\n=======CLONES========\n")
        # print(f"\n{clones}\n")
        print(f"\n{len(clones)}\n")


     
# run_ci_pipeline()
if __name__ == '__main__':
    run_ci_pipeline()


