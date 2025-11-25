import os


def replace_imports_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Replace only the exact prefix
                new_content = content.replace("from mlfinlab", "from .")
                if new_content != content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"Updated: {file_path}")


# Example usage:
# replace_imports_in_folder("/path/to/your/folder")
