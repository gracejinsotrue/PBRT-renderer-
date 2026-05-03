import re

with open(r'C:\Users\gjin3\Desktop\nori-26sp\scenes\final_scene_real\scene.xml', 'r', encoding='utf-8') as f:
    content = f.read()

comments = [(m.start(), m.end()) for m in re.finditer(r'<!--.*?-->', content, re.DOTALL)]

def is_in_comment(pos):
    for s, e in comments:
        if s <= pos < e:
            return True
    return False

active_meshes = []
for m in re.finditer(r'<mesh\b', content):
    if not is_in_comment(m.start()):
        line = content[:m.start()].count('\n') + 1
        snippet = content[m.start():m.start()+300]
        fn = re.search(r'filename[^"]*"([^"]*)"', snippet)
        active_meshes.append((line, fn.group(1) if fn else '???'))

print(f'Total active meshes: {len(active_meshes)}')
for line, fn in active_meshes:
    print(f'  line {line}: {fn}')
