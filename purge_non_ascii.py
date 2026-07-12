import os
import re

def purge_non_ascii(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return False
    
    # Common replacements
    replacements = {
        '-': '-',
        '-': '-',
        '->': '->',
        '=': '=',
        '-': '-',
        '...': '...',
        '*': '*',
        'Rs.': 'Rs.',
        '(TM)': '(TM)',
        '(C)': '(C)',
        '(R)': '(R)',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        'sigma': 'sigma',
        'Sum': 'Sum',
        'sqrt': 'sqrt',
        'x': 'x',
        '>=': '>=',
        '<=': '<=',
        '+/-': '+/-',
        '!=': '!=',
        '~': '~',
        'inf': 'inf',
        'Delta': 'Delta',
        'mu': 'mu',
    }
    
    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
    
    # Remove any remaining non-ASCII
    new_content = re.sub(r'[^\x00-\x7f]', '', new_content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

root_path = r"c:\Users\stuti\OneDrive\Desktop\MSC_Project"
targets = [
    os.path.join(root_path, "src"),
    os.path.join(root_path, "configs"),
    root_path
]

for target in targets:
    if os.path.isfile(target):
        if target.endswith('.py') or target.endswith('.md'):
            if purge_non_ascii(target):
                print(f"Purged: {target}")
    else:
        for item in os.listdir(target):
            path = os.path.join(target, item)
            if os.path.isfile(path):
                if path.endswith('.py') or path.endswith('.md'):
                    if purge_non_ascii(path):
                        print(f"Purged: {path}")
            elif os.path.isdir(path) and item in ["src", "configs"]:
                for r, d, f in os.walk(path):
                    for file in f:
                        if file.endswith('.py') or file.endswith('.md'):
                            p = os.path.join(r, file)
                            if purge_non_ascii(p):
                                print(f"Purged: {p}")
