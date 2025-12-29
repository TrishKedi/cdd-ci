from core.services import CodeAnalyzer

code_analyzer = CodeAnalyzer()

def run_ci_pipeline():
    changed_files = code_analyzer.get_changed_files()
    print(changed_files)


# run_ci_pipeline()
if __name__ == '__main__':
    run_ci_pipeline()
