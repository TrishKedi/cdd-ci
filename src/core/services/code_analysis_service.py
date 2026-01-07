import re
import argparse
import subprocess
import tree_sitter_javascript as tsjs
from pathlib import Path
from typing import List, Iterator, Dict, Any, Tuple, Optional
from tree_sitter import Language, Parser, Node, Tree, Query
from multiprocessing import Pool, cpu_count


EXCLUDE_PATTERN = [
    '**.test.js'
]

MINIFIED_INDICATORS = [
    r'sourceMappingURL',
    r'\.min\.js$',
    r'^[^\\n]{500,}$',
    r'\]{3,}',
    r'\}{3,}'
    
]

class CodeAnalyzer:

    def __init__(self):
        self.exclude_pattern: List[str] = EXCLUDE_PATTERN
        self.minified_patterns: List[re.Pattern[str]] = [
            re.compile(pattern) for pattern in MINIFIED_INDICATORS
        ]
        self.max_size: int = 1000000
        self.controller_query = r"""
            (call_expression
            function: (member_expression
                property: (property_identifier) @callee_name
            )
            arguments: (arguments
                (_)
                (object) @controller_obj
                .
            )
            (#eq? @callee_name "extend")
            )"""

        self.functions_query = r"""
        (object 
        (pair 
            key: (property_identifier) @method_name
            value: (arrow_function) @afn
        )
        )
      
        (object
        (method_definition
            name: (property_identifier) @method_name
        )
        ) @afn

        (pair
            key: (property_identifier) @method_name
            value: (function_expression) @afn
        )

        (method_definition
            name: (property_identifier) @method_name
        ) @afn
        """

    def get_changed_files(self):
        print("Get changed files")
        parser = argparse.ArgumentParser()
        parser.add_argument('--changed-files', required=True)
        args = parser.parse_args()

        with open(args.changed_files, 'r', encoding='utf-8') as f:
            files = f.read()
            changed_files = files.split('\n')

            return changed_files
            # print(files.split('\n'))

    # def get_changed_files(self):
    #     changed_files = []
    #     try:
    #         print("retrieving changed files")
    #         diff_result = subprocess.run(
    #             ['git', 'diff', '--name-only', '*.js'],
    #             capture_output=True,
    #             text=True,
    #             timeout=300
    #         )

    #         if diff_result.returncode == 0:
    #             raw_output = diff_result.stdout
    #             output =  raw_output.split("\n")
    #             changed_files = output

    #     except subprocess.TimeoutExpired:
    #         print(f"Timed out")

    #     except Exception as e:
    #         print(f"Failed to get changed files: {e}")

    #     finally:
    #         return changed_files

    def should_process_file(self, file_path: Path) -> bool:

        try:
            if not self._passes_exclude_pattern(file_path):
                return False

            if not self._passes_minification_check(file_path):
                return False

            if not self._passes_size_check(file_path):
                return False

            return True

        except Exception as e:
            print(f"Failed to check file {e}")
            return False

    def _passes_exclude_pattern(self, file_path: Path) -> bool:

        for pattern in self.exclude_pattern:
            if file_path.match(pattern):
                return False

            if pattern.startswith('**/') and pattern.endswith('/**'):
                folder_name = pattern[3:-3]
                if folder_name in file_path.parts:
                    return False

        return True

    def _passes_minification_check(self, file_path: Path) -> bool:
        sample = file_path.read_text(encoding='utf-8', errors='ignore')[:5000]

        for pattern in self.minified_patterns:
            if pattern.search(sample):
                return False

        max_line_length = max((len(line) for line in sample.splitlines()[:10]), default=0)

        return max_line_length <= 500

    def _passes_size_check(self, file_path: Path) -> bool:

        return file_path.stat().st_size <= self.max_size
     

    def _process_single_file(self, file_path: Path) -> List[Dict[str, Any]]:

        try:
            fragments = self.fragments_from_file(file_path)
            return fragments

        except Exception as e:
            print(f'Failed to extract Fragments: {e}')
            return []

    def get_controller_node(self, tree: Tree, language: Language) -> Node:
        controller_q = Query(language, self.controller_query)
        controller_q_result = controller_q.captures(tree.root_node)

        controller_objs = controller_q_result.get('controller_obj', [])

        return controller_objs if isinstance(controller_objs, list) else [controller_objs]


    def extract_function_nodes(self, root_node: Node, code_bytes: bytes, language: Language) -> List[Dict[str, Any]]:
        query = Query(language, self.functions_query)

        function_types = {
            "function_declaration",
            "function_expression",
            "arrow_function",
            "method_definition"
        }

        fragments: List[Dict[str, Any]] = []
        func_captures = query.captures(root_node)
        nodes = func_captures.get('afn', [])

        if not isinstance(nodes, list):
            nodes = [nodes] if node else []

        for node in nodes:
            if node.type in function_types:
                start_byte,  end_byte = node.start_byte, node.end_byte
                fragment_code = code_bytes[start_byte:end_byte].decode("utf-8", errors="igonre")

                start_line = code_bytes[:start_byte].count(b'\n') + 1
                end_line = code_bytes[:end_byte].count(b'\n') + 1

                fragments.append({
                    "code": fragment_code,
                    "start": start_line,
                    "end": end_line
                })

        return fragments

    def normalize_js(self, snippet: str) -> str:
        space = re.compile(r"\s+")

        snippet = re.sub(r"//.*$", " ", snippet, flags=re.M)
        snippet = re.sub(r"/\*.*?\*/", " ", snippet, flags=re.S)

        snippet = snippet.replace(";", "")
        snippet = space.sub("", snippet).strip()

        return snippet

    def fragments_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        
        language = Language(tsjs.language())

        parser = Parser(language)

        code = file_path.read_text(encoding='utf-8')
        tree = parser.parse(bytes(code, 'utf-8'))
        root_node = tree.root_node
        ctr_node = self.get_controller_node(tree, language)

        if ctr_node:
            root_node = ctr_node[0]
        
        code_bytes = code.encode('utf-8')
        fragments = self.extract_function_nodes(root_node, code_bytes, language)

        processed_fragments = [
            {
                **fragment,
                "processedCode": self.normalize_js(fragment.get('code', '')),
                "path": str(file_path)

            }
            for fragment in fragments
        ]

        return processed_fragments

    def extract_candidate_files(self, code_dir:str)->List[Path]:
        directory = Path(code_dir)
        
        if not directory.exists() or not directory.is_dir():
            print(f"❌ {directory} is not a valid directory.")
            # raise typer.Exit(code=1)
            return

        # Get list of all JS files and filter them
        return list(directory.rglob("*.js"))

    def extract_code_chunks(self, query_files:bool=True, code_dir:str="repos", batch_size:int=50 ) -> Iterator[Dict[str, Any]]:

        files = self.get_changed_files()
        print(f"Files: {files}")

        if not query_files:
            files = self.extract_candidate_files(code_dir)
        
        
        if files:
            
            valid_files = [
                Path(file_path) for file_path in files 
                if file_path  and self.should_process_file(Path(file_path))
            ]

            if valid_files:
                print(f"File paths: {valid_files}")
                num_processes = min(cpu_count(), len(valid_files))
                
                chunk_size = max(1, (len(valid_files) // num_processes))
                print(f"chunk_size: {chunk_size}")
                with Pool(processes=num_processes) as pool:
                    chunks = pool.imap(self._process_single_file, valid_files, chunksize=50)
                    
                 
                    for chunk in chunks:
                        # print(len(chunk))
                        for i in range(0, len(chunk), batch_size):
                            # print(i)
                            chunk_batch = chunk[i:i+batch_size]
                            yield  chunk_batch
                            # print(f"\n{chunk_batch}\n")
                        
                        # for code_chunk in chunk:
                        #     print(f"\n{code_chunk}\n")
                        #     # yield code_chunk

                

          

    def get_processed_code(self, chunks, all=False) -> List[str]:

        code_chunks = []
        for file_chunks in chunks:
        
            for chunk in file_chunks:
                # print(chunk)
                not all and code_chunks.append(chunk.get('processedCode'))
                all and code_chunks.append(chunk)

        return code_chunks
        # def file_chunks(f_chunks):
        #     return [chunk.get('processedCode') for chunk in f_chunks]

        # return list(map(file_chunks, chunks))


