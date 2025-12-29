from core.services import CodeAnalyzer

code_analyzer = CodeAnalyzer()

def run_ci_pipeline():

    for chunk in code_analyzer.extract_code_chuncks():
        
        print(f"\n{chunk}\n")
     
# run_ci_pipeline()
if __name__ == '__main__':
    run_ci_pipeline()
