import os
import re
import argparse
import shutil


def delete_pycache(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if __pycache__ is in the current directory
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            try:
                shutil.rmtree(pycache_path)  # Recursively delete the directory
                print(f"Deleted: {pycache_path}")
            except Exception as e:
                print(f"Failed to delete {pycache_path}: {e}")

def replace_keyword_in_files(root_dir, keyword, replacement):
    # Compile the keyword for performance if using regular expressions
    keyword_pattern = re.compile(re.escape(keyword))  # Escapes special regex characters in keyword
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # if "__pycache__" in dirpath:
        #     continue
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Replace all occurrences of the keyword
                new_content = keyword_pattern.sub(replacement, content)
                
                # Write back the modified content only if changes were made
                if content != new_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Modified: {file_path}")
                    # exit(0)
            
            except (UnicodeDecodeError, PermissionError, FileNotFoundError) as e:
                print(f"Skipping file {file_path} due to error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace all occurrences of a keyword in files under a directory.")
    parser.add_argument("-k", "--target_keyword", type=str, help="The keyword to search for.")
    parser.add_argument("-r", "--replacement_string", type=str, help="The string to replace the keyword with.")
    parser.add_argument("-d", "--root_directory", type=str, help="The root directory to search in.")

    args = parser.parse_args()

    target_keyword = args.target_keyword
    replacement_string = args.replacement_string
    root_directory = args.root_directory

    delete_pycache(root_directory)
    replace_keyword_in_files(root_directory, target_keyword, replacement_string)
