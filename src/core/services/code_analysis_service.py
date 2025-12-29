import subprocess

class CodeAnalyzer:

    def get_changed_files(self):
        try:
            print("retrieving changed files")
            changed_files = subprocess.run(['git', 'diff', '--name-only', '*.py'])

        except Exception as e:
            print(f"Failed to get changed files: {e}")

    def extract_code_chuncks():
        pass
